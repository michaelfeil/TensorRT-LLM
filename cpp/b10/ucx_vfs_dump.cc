#include "b10/ucx_vfs_dump.h"

#include "tensorrt_llm/common/logger.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C"
{
#include <ucs/datastruct/string_buffer.h>
#include <ucs/vfs/base/vfs_obj.h>
}

namespace b10
{

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

namespace
{

constexpr char kDumpDir[] = "/tmp/ucx_vfs";
constexpr int kPollIntervalSec = 5;

// ---------------------------------------------------------------------------
// Helpers — JSON escape, list-dir callback
// ---------------------------------------------------------------------------

struct ListDirCtx
{
    std::vector<std::string> names;
};

void ListDirCb(char const* name, void* arg)
{
    static_cast<ListDirCtx*>(arg)->names.emplace_back(name);
}

// JSON-escape the minimal set of characters that break parsing. VFS file
// contents are short ASCII (counters, transport names); minimal is enough.
std::string JsonEscape(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 8);

    for (char c : s)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            }
            else
            {
                out += c;
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Walker — recursive tree → JSON
// ---------------------------------------------------------------------------

void WalkJson(std::string const& vfs_path, std::ostream& out, int indent)
{
    // libucs has no node for the root path "/" — get_info would return
    // UCS_ERR_NO_ELEM. list_dir has a special case for "/" that walks the
    // implicit root, so treat root as a directory unconditionally.
    bool const is_root = (vfs_path == "/");
    ucs_vfs_path_info_t info{};
    if (!is_root && ucs_vfs_path_get_info(vfs_path.c_str(), &info) != UCS_OK)
    {
        out << "null";
        return;
    }

    const std::string pad(indent, ' ');
    const std::string pad_inner(indent + 2, ' ');

    if (is_root || S_ISDIR(info.mode))
    {
        ListDirCtx ctx;
        ucs_vfs_path_list_dir(vfs_path.c_str(), ListDirCb, &ctx);

        out << "{\n";
        for (size_t i = 0; i < ctx.names.size(); ++i)
        {
            auto const& name = ctx.names[i];
            const std::string child = (vfs_path == "/") ? ("/" + name) : (vfs_path + "/" + name);

            out << pad_inner << "\"" << JsonEscape(name) << "\": ";
            WalkJson(child, out, indent + 2);

            if (i + 1 < ctx.names.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << pad << "}";
        return;
    }

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    if (ucs_vfs_path_read_file(vfs_path.c_str(), &strb) == UCS_OK)
    {
        char const* raw = ucs_string_buffer_cstr(&strb);
        std::string content(raw ? raw : "");
        // libucs file contents end in '\n'; strip for cleaner JSON.
        while (!content.empty() && content.back() == '\n')
            content.pop_back();
        out << "\"" << JsonEscape(content) << "\"";
    }
    else
    {
        out << "null";
    }
    ucs_string_buffer_cleanup(&strb);
}

// ---------------------------------------------------------------------------
// Serializer — walk + atomic-rename write
// ---------------------------------------------------------------------------

void WriteJsonFile(std::string const& path, int rank)
{
    const std::string tmp = path + ".tmp";

    std::ofstream out(tmp, std::ios::trunc);
    if (!out)
    {
        TLLM_LOG_WARNING(rank, "UcxVfsDump: cannot open %s for write (errno=%d)", tmp.c_str(), errno);
        return;
    }

    out << "{\n";
    out << "  \"pid\": " << static_cast<long>(::getpid()) << ",\n";
    out << "  \"rank\": " << rank << ",\n";
    out << "  \"tree\": ";
    WalkJson("/", out, 2);
    out << "\n}\n";
    out.close();

    if (std::rename(tmp.c_str(), path.c_str()) != 0)
    {
        TLLM_LOG_WARNING(rank, "UcxVfsDump: rename %s -> %s failed (errno=%d)", tmp.c_str(), path.c_str(), errno);
        std::remove(tmp.c_str());
    }
}

bool DumpDirExists()
{
    struct stat st
    {
    };

    return ::stat(kDumpDir, &st) == 0 && S_ISDIR(st.st_mode);
}

std::string RankFilePath(int rank)
{
    return std::string(kDumpDir) + "/rank-" + std::to_string(rank) + ".json";
}

// ---------------------------------------------------------------------------
// Dumper — owns the thread; idle until the dump dir exists
// ---------------------------------------------------------------------------

class Dumper
{
public:
    explicit Dumper(int rank)
        : rank_(rank)
    {
        thread_ = std::thread([this] { this->Run(); });
    }

    ~Dumper()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable())
        {
            thread_.join();
        }
    }

private:
    void Run()
    {
        // Toggle: operator creates the dump dir to start dumps, removes it to
        // stop. State transitions log once each so worker stdout reflects what's
        // happening; steady-state polls are silent.
        bool was_active = false;

        while (!stop_.load(std::memory_order_relaxed))
        {
            bool const is_active = DumpDirExists();

            if (is_active && !was_active)
            {
                TLLM_LOG_INFO(rank_, "UcxVfsDump: %s exists, dumping every %ds", kDumpDir, kPollIntervalSec);
                was_active = true;
            }
            else if (!is_active && was_active)
            {
                TLLM_LOG_INFO(rank_, "UcxVfsDump: %s gone, going idle", kDumpDir);
                was_active = false;
            }

            if (is_active)
            {
                try
                {
                    WriteJsonFile(RankFilePath(rank_), rank_);
                }
                catch (std::exception const& e)
                {
                    TLLM_LOG_WARNING(rank_, "UcxVfsDump: dump failed: %s", e.what());
                }
            }

            // Sleep in 1-second slices so the destructor unblocks within ~1s.
            for (int i = 0; i < kPollIntervalSec && !stop_.load(std::memory_order_relaxed); ++i)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }

    int rank_;
    std::atomic<bool> stop_{false};
    std::thread thread_;

    Dumper(Dumper const&) = delete;
    Dumper& operator=(Dumper const&) = delete;
};

std::mutex singleton_mu_;
std::unique_ptr<Dumper> singleton_;

} // namespace

// ---------------------------------------------------------------------------
// Public hook surface
// ---------------------------------------------------------------------------

void StartUcxStat(int rank)
{
    std::lock_guard<std::mutex> lock(singleton_mu_);
    if (singleton_)
    {
        return; // already started; subsequent calls are no-ops
    }
    singleton_ = std::make_unique<Dumper>(rank);
}

void StopUcxStat()
{
    std::lock_guard<std::mutex> lock(singleton_mu_);
    singleton_.reset();
}

} // namespace b10
