#pragma once

// ---------------------------------------------------------------------------
// In-process UCX VFS state dumper.
//
// Polls the libucs vfs_obj tree via the public ucs_vfs_path_* API and
// writes JSON snapshots to disk on a periodic interval. Same view as the
// FUSE mount at /tmp/ucx/<pid>/, but no CAP_SYS_ADMIN, no /dev/fuse, no
// unconfined AppArmor — the privileges that block FUSE in production-shaped
// pods.
//
// Works against stock libucs — the base vfs_obj tree at src/ucs/vfs/base/
// is built into libucs.so unconditionally and UCP/UCT objects register
// into it unconditionally. No --enable-vfs rebuild, no UCX_VFS_ENABLE env,
// no daemon.
//
// The dumper thread starts unconditionally from UcxConnectionManager's
// constructor and stays idle until /tmp/ucx_vfs/ exists. Operators toggle
// dumps on/off with `mkdir /tmp/ucx_vfs` / `rm -rf /tmp/ucx_vfs` — no env
// vars to inject, no worker restart. Each rank writes its own file:
// /tmp/ucx_vfs/rank-<rank>.json, refreshed every 5s.
//
// Design: docs/transfer/ucx_observability_no_fuse_design.md
// FUSE counterpart: docs/transfer/ucx_observability.md
// ---------------------------------------------------------------------------

namespace b10
{

// Hook entry point: start the dumper thread unconditionally.
// Safe to call multiple times; only the first call starts a thread.
// Called from UcxConnectionManager's constructor after ucxx::createContext().
void StartUcxStat(int rank);

// Stop and join the dumper thread if it was started.
// Idempotent. Must be called before libucs teardown.
// Called from UcxConnectionManager's destructor.
void StopUcxStat();

} // namespace b10
