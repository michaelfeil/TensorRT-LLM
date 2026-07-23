import asyncio
import hashlib
import hmac
import os
import pickle  # nosec B403
import threading
import time
import traceback
from queue import Queue
from typing import Any, Iterable, Optional

import zmq
import zmq.asyncio
from blake3 import blake3

from tensorrt_llm.logger import logger

from .._utils import nvtx_mark, nvtx_range_debug
from ..llmapi.utils import (ManagedThread, enable_llm_debug, logger_debug,
                            print_colored)

_AUTH_TAG_BYTES = hashlib.sha256().digest_size
_BLAKE3_KEY_DOMAIN = b"TensorRT-LLM IPC BLAKE3 key v1\0"
_BLAKE3_MAX_THREADS = 4
_BLAKE3_PARALLEL_MIN_FRAME_BYTES = 4 * 1024 * 1024


class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
        zmq.ROUTER: "ROUTER",
        zmq.DEALER: "DEALER",
    }

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 use_hmac_encryption: bool = True):
        '''
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address and authentication key for the IPC.
                Defaults to None.
            socket_type (int): The type of socket to use. Defaults to zmq.PAIR.
            is_server (bool): Whether the current process is the server or the client.
            is_async (bool): Whether to use asyncio for the socket. Defaults to False.
            name (str, optional): The name of the queue. Defaults to None.
            use_hmac_encryption (bool): Legacy name for mandatory message authentication. Single-frame
                messages use HMAC-SHA256 and multipart messages use keyed BLAKE3. Defaults to True.
        '''

        assert use_hmac_encryption, "Message authentication is required to prevent unauthorized deserialization."

        self.socket_type = socket_type
        self.address_endpoint = address[
            0] if address is not None else "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket = self.context.socket(socket_type)
        self.socket.set_hwm(0)

        # For ROUTER sockets, track the last identity to enable replies. For now we assume there is only one client in our case.
        self._last_identity = None

        self.hmac_key = address[1] if address is not None else None
        self.use_hmac_encryption = use_hmac_encryption
        self._hmac_proto = None
        self._hmac_key_cached = None
        self._blake3_key = None
        self._blake3_key_source = None

        self._setup_lock = threading.Lock()

        # Thread safety debugging
        self._zmq_thread_id = None
        self._zmq_debug_enabled = os.environ.get('TLLM_LLMAPI_ZMQ_DEBUG',
                                                 '0') != '0'

        # Check HMAC key condition
        if self.use_hmac_encryption and not self.is_server and self.hmac_key is None:
            raise ValueError(
                "Client must receive HMAC key when encryption is enabled")
        elif not self.use_hmac_encryption and self.hmac_key is not None:
            raise ValueError(
                "Server and client should not receive HMAC key when encryption is disabled"
            )

        if self.should_bind_socket():
            self.socket.bind(
                self.address_endpoint
            )  # Binds to the address and occupy a port immediately
            self.address_endpoint = self.socket.getsockopt(
                zmq.LAST_ENDPOINT).decode()
            logger_debug(
                f"Server [{name}] bound to {self.address_endpoint} in {self.socket_type_str[socket_type]}\n",
                "green")

            if self.use_hmac_encryption and not self.hmac_key:
                # Initialize HMAC key for pickle encryption
                logger.info(f"Generating a new HMAC key for server {self.name}")
                self.hmac_key = os.urandom(32)

            self.address = (self.address_endpoint, self.hmac_key)

    def should_bind_socket(self) -> bool:
        """
        Determine if socket should bind vs connect based on type and role.

        ZMQ binding conventions:
        - PAIR: server binds, client connects (1-to-1 bidirectional)
        - PULL: server binds to receive from multiple PUSH sockets
        - PUSH: server binds when acting as message source
        - ROUTER: always binds to handle multiple clients

        Returns:
            True if socket should bind, False if it should connect
        """
        # Server binds for PAIR, PULL, PUSH patterns
        if self.is_server and self.socket_type in (zmq.PAIR, zmq.PULL,
                                                   zmq.PUSH):
            return True

        # ROUTER always binds (multi-client pattern)
        if self.socket_type == zmq.ROUTER:
            return True

        # Client connects for all other cases
        return False

    def setup_lazily(self):
        # Early return if setup is already done
        if self._setup_done:
            return

        with self._setup_lock:
            if self._setup_done:
                return
            self._setup_done = True

            if not self.is_server:
                logger_debug(
                    f"Client [{self.name}] connecting to {self.address_endpoint} in {self.socket_type_str[self.socket_type]}\n",
                    "green")
                self.socket.connect(self.address_endpoint)

            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)

    def _check_thread_safety(self):
        """Check if the current thread is the same as the thread that first used the socket."""
        if not self._zmq_debug_enabled:
            return

        current_thread_id = threading.get_ident()

        if self._zmq_thread_id is None:
            # First call - capture the thread ID
            self._zmq_thread_id = current_thread_id
            logger_debug(
                f"ZMQ socket [{self.name}] initialized on thread {current_thread_id}",
                "cyan")
        elif self._zmq_thread_id != current_thread_id:
            # Thread mismatch - raise error
            raise RuntimeError(
                f"ZMQ thread safety violation detected in [{self.name}]: "
                f"Socket created on thread {self._zmq_thread_id}, "
                f"but accessed from thread {current_thread_id}. "
                f"ZMQ sockets are not thread-safe!")

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()
        self._check_thread_safety()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def put(self, obj: Any, routing_id: Optional[bytes] = None):
        self.setup_lazily()
        self._check_thread_safety()
        with nvtx_range_debug("send", color="blue", category="IPC"):
            if self.socket_type in (zmq.ROUTER, zmq.DEALER):
                data = self._prepare_data(obj)
                self._send_data(data, routing_id=routing_id)
            elif self.use_hmac_encryption:
                # Preserve put()'s snapshot semantics.
                self.socket.send_multipart(self._prepare_frames(obj), copy=True)
            else:
                self.socket.send_pyobj(obj)

    def put_noblock(self,
                    obj: Any,
                    *,
                    retry: int = 1,
                    wait_time: float = 0.001):
        '''
        Put an object into the queue without blocking, and retry if the send fails.
        NOTE: It won't raise any error if the send fails.

        Parameters:
            obj (Any): The object to send.
            retry (int): The number of times to retry sending the object.
            wait_time (float): The time to wait before retrying.
        '''

        assert retry >= 0 and retry <= 10, "Retry must be between 0 and 10, adjust the wait_time if needed"

        self.setup_lazily()
        self._check_thread_safety()
        with nvtx_range_debug("send", color="blue", category="IPC"):

            data = self._prepare_data(obj)
            try:
                self._send_data(data, flags=zmq.NOBLOCK)
            except zmq.Again:
                if retry > 0:
                    time.sleep(wait_time)
                    self.put_noblock(obj, retry=retry - 1, wait_time=wait_time)
                else:
                    logger.error(f"Failed to send object: {obj}")

    async def put_async(self, obj: Any, routing_id: Optional[bytes] = None):
        self.setup_lazily()
        self._check_thread_safety()
        try:
            if self.socket_type in (zmq.ROUTER, zmq.DEALER):
                data = self._prepare_data(obj)
                await self._send_data_async(data, routing_id=routing_id)
            elif self.use_hmac_encryption:
                await self.socket.send_multipart(self._prepare_frames(obj),
                                                 copy=True)
            else:
                await self.socket.send_pyobj(obj)
        except TypeError as e:
            logger.error(f"Cannot pickle {obj}")
            raise e
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

        nvtx_mark("ipc.send", color="blue", category="IPC")

    async def put_async_noblock(self, obj: Any):
        self.setup_lazily()
        self._check_thread_safety()
        try:
            if self.socket_type in (zmq.ROUTER, zmq.DEALER):
                await self.socket.send(self._prepare_data(obj),
                                       flags=zmq.NOBLOCK)
            elif self.use_hmac_encryption:
                await self.socket.send_multipart(self._prepare_frames(obj),
                                                 flags=zmq.NOBLOCK,
                                                 copy=True)
            else:
                await self.socket.send_pyobj(obj, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

    def get(self) -> Any:
        self.setup_lazily()
        self._check_thread_safety()
        return self._recv_data()

    def drain(self) -> list[Any]:
        """Non-blocking drain: return all currently available messages without waiting."""
        self.setup_lazily()
        self._check_thread_safety()
        results = []
        while self.socket.poll(timeout=0):
            results.append(self._recv_data())
        return results

    async def get_async(self) -> Any:
        self.setup_lazily()
        self._check_thread_safety()
        return await self._recv_data_async()

    async def get_async_noblock(self,
                                timeout: float = 0.5,
                                return_identity: bool = False) -> Any:
        """Get data with timeout using polling to avoid message drops.

        This method uses ZMQ's NOBLOCK flag with polling instead of asyncio.wait_for
        to prevent cancelling recv operations which can cause message drops.

        Args:
            timeout: Timeout in seconds
            return_identity: Whether to return the identity of the sender (for ROUTER sockets)

        Returns:
            The received object, or (object, identity) if return_identity is True

        Raises:
            asyncio.TimeoutError: If timeout is reached without receiving data
        """
        self.setup_lazily()
        self._check_thread_safety()

        # Use polling loop instead of asyncio.wait_for to avoid cancelling recv
        # which can cause message drops
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                # Try non-blocking receive
                if self.socket_type == zmq.ROUTER:
                    identity, data = await self.socket.recv_multipart(
                        flags=zmq.NOBLOCK)
                    self._last_identity = identity
                    obj = self._parse_data(data)
                    if return_identity:
                        return obj, identity
                    else:
                        return obj
                else:
                    if self.use_hmac_encryption:
                        frames = await self.socket.recv_multipart(
                            flags=zmq.NOBLOCK, copy=False)
                        obj = self._parse_frames(frames)
                    else:
                        obj = await self.socket.recv_pyobj(flags=zmq.NOBLOCK)

                    if return_identity:
                        return obj, None
                    else:
                        return obj
            except zmq.Again:
                # No message available yet
                remaining_ms = int(
                    (deadline - asyncio.get_event_loop().time()) * 1000)
                if remaining_ms <= 0:
                    raise asyncio.TimeoutError()
                # Use async poller to wait for data without busy-polling on a
                # fixed interval, which avoids latency spikes from sleep(0.01)
                async_poller = zmq.asyncio.Poller()
                async_poller.register(self.socket, zmq.POLLIN)
                try:
                    events = await async_poller.poll(timeout=remaining_ms)
                finally:
                    async_poller.unregister(self.socket)
                if not events:
                    raise asyncio.TimeoutError()

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _keyed_hmac(self) -> "hmac.HMAC":
        """Return a copy of the cached keyed HMAC state."""
        if self._hmac_key_cached is not self.hmac_key:
            self._hmac_proto = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
            self._hmac_key_cached = self.hmac_key
        return self._hmac_proto.copy()

    def _frame_auth_tag(self, frames: Iterable[bytes | memoryview]) -> bytes:
        """Authenticate multipart frames with keyed BLAKE3.

        BLAKE3 is not FIPS-approved. Peers must use the same multipart
        authentication scheme because the wire format has no negotiation.
        """
        views = [memoryview(frame) for frame in frames]
        hmac_key = self.hmac_key
        if hmac_key is None:
            raise RuntimeError("HMAC key is not initialized")

        if self._blake3_key_source is not hmac_key:
            self._blake3_key = hmac.digest(hmac_key, _BLAKE3_KEY_DOMAIN,
                                           "sha256")
            self._blake3_key_source = hmac_key
        largest_frame_bytes = max((view.nbytes for view in views), default=0)
        max_threads = (_BLAKE3_MAX_THREADS if largest_frame_bytes
                       >= _BLAKE3_PARALLEL_MIN_FRAME_BYTES else 1)
        h = blake3(key=self._blake3_key, max_threads=max_threads)
        for frame in views:
            h.update(frame.nbytes.to_bytes(8, "little"))
            h.update(frame)
        return h.digest()

    def _verify_hmac(self, data: bytes | memoryview,
                     actual_hmac: bytes | memoryview) -> bool:
        """Verify the HMAC of received pickle data."""
        h = self._keyed_hmac()
        h.update(data)
        return hmac.compare_digest(h.digest(), actual_hmac)

    def _sign_data(self, data_before_encoding: bytes) -> bytes:
        """Generate HMAC for data."""
        h = self._keyed_hmac()
        h.update(data_before_encoding)
        return data_before_encoding + h.digest()

    def __del__(self):
        self.close()

    def _prepare_data(self, obj: Any) -> bytes:
        """Serialize object and optionally add HMAC signature."""
        data = pickle.dumps(obj)  # nosec B301
        if self.use_hmac_encryption:
            return self._sign_data(data)
        return data

    def _prepare_frames(self, obj: Any) -> list:
        """Serialize an object as authenticated pickle-5 frames."""
        from tensorrt_llm._torch.distributed.communicator import \
            _dumps_with_oob_buffers
        metadata, buffers = _dumps_with_oob_buffers(obj)
        if not buffers:
            return [self._sign_data(bytes(metadata))]
        tag = self._frame_auth_tag((metadata, *buffers))
        return [bytes(metadata) + tag, *buffers]

    def _parse_frames(self, frames: list) -> Any:
        """Deserialize frames created by _prepare_frames."""
        if len(frames) == 1:
            return self._parse_data(frames[0].buffer)

        views = [frame.buffer for frame in frames]
        metadata = views[0][:-_AUTH_TAG_BYTES]
        actual_tag = views[0][-_AUTH_TAG_BYTES:]
        if not hmac.compare_digest(self._frame_auth_tag(
            (metadata, *views[1:])), actual_tag):
            raise RuntimeError("Message authentication failed")
        # Rebuilt tensors retain the ZMQ frames through their buffer views.
        return pickle.loads(metadata, buffers=views[1:])  # nosec B301

    def _parse_data(self, data: bytes | memoryview) -> Any:
        """Deserialize data after verifying its optional HMAC signature."""
        if not self.use_hmac_encryption:
            return pickle.loads(data)  # nosec B301

        view = memoryview(data)
        message_data = view[:-_AUTH_TAG_BYTES]
        actual_hmac = view[-_AUTH_TAG_BYTES:]
        if not self._verify_hmac(message_data, actual_hmac):
            raise RuntimeError("HMAC verification failed")
        return pickle.loads(message_data)  # nosec B301

    def _send_data(self,
                   data: bytes,
                   flags: int = 0,
                   routing_id: Optional[bytes] = None):
        """Send data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            identity = routing_id if routing_id is not None else self._last_identity
            if identity is None:
                raise ValueError("ROUTER socket requires identity")
            self.socket.send_multipart([identity, data], flags=flags)
        else:
            self.socket.send(data, flags=flags)

    async def _send_data_async(self,
                               data: bytes,
                               routing_id: Optional[bytes] = None):
        """Async version of _send_data."""
        if self.socket_type == zmq.ROUTER:
            identity = routing_id if routing_id is not None else self._last_identity
            if identity is None:
                raise ValueError("ROUTER socket requires identity")
            await self.socket.send_multipart([identity, data])
        else:
            await self.socket.send(data)

    def _recv_data(self, return_identity: bool = False) -> Any:
        """Receive data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            identity, data = self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            obj = self._parse_data(data)
            if return_identity:
                return obj, identity
            return obj
        else:
            if self.use_hmac_encryption:
                frames = self.socket.recv_multipart(copy=False)
                obj = self._parse_frames(frames)
            else:
                obj = self.socket.recv_pyobj()

            if return_identity:
                return obj, None
            return obj

    async def _recv_data_async(self, return_identity: bool = False) -> Any:
        """Async version of _recv_data."""
        if self.socket_type == zmq.ROUTER:
            identity, data = await self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            obj = self._parse_data(data)
            if return_identity:
                return obj, identity
            return obj
        else:
            if self.use_hmac_encryption:
                frames = await self.socket.recv_multipart(copy=False)
                obj = self._parse_frames(frames)
            else:
                obj = await self.socket.recv_pyobj()

            if return_identity:
                return obj, None
            return obj

    def notify_with_retry(self, message, max_retries=5, timeout=1):
        """
        Notify with automatic retry on failure (for DEALER socket pattern).

        Args:
            message: Message to send
            max_retries: Maximum retry attempts (default: 5)
            timeout: Timeout in seconds for each attempt (default: 1)

        Returns:
            bool: True if acknowledgment received, False if failed after all retries
        """
        if self.socket_type != zmq.DEALER:
            raise ValueError(
                "notify_with_retry is only supported for DEALER socket for now")

        self._check_thread_safety()
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.put(message)
                # Wait for ACK with timeout
                if self.poll(timeout):
                    self.get()
                    return True
                else:
                    retry_count += 1

            except Exception as e:
                logger.error(f"Failed to notify with retry: {e}")
                retry_count += 1

        return False


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(obj)
        else:
            batch = obj if isinstance(obj, list) else [obj]
            self.queue.put(batch)

    def get(self) -> Any:
        return self.queue.get()

    @property
    def address(self) -> tuple[str, Optional[bytes]]:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
