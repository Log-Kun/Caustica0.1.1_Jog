package dev.comfyfluffy.caustica.rt.overlay;

import org.lwjgl.vulkan.VK10;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;

/**
 * Per-frame host-visible vertex/index scratch for overlay passes, shared by every {@link RtOverlayFeature}.
 * Buffers acquired during a frame are queued at {@link #endFrame} and destroyed only {@value #KEEP_FRAMES}
 * frames later (the same frames-in-flight-safe deferred-release convention {@code RtEntities} uses), so a
 * buffer is never destroyed while a prior frame's GPU reads are still in flight.
 */
public final class RtOverlayFramePool {
    private static final int KEEP_FRAMES = 4;
    // Vulkan requires buffer size > 0; a few zero-length overlay draws could otherwise reach acquire() with
    // bytes == 0.
    private static final long MIN_SIZE = 256;

    private final List<RtBuffer> acquiredThisFrame = new ArrayList<>();
    private final List<Deferred> deferred = new ArrayList<>();

    private record Deferred(long freeFrame, RtBuffer buffer) {
    }

    /** Destroy buffers whose in-flight window has passed. Call once at the start of the overlay frame. */
    public void beginFrame(long frameCounter) {
        Iterator<Deferred> it = deferred.iterator();
        while (it.hasNext()) {
            Deferred d = it.next();
            if (d.freeFrame <= frameCounter) {
                d.buffer.destroy();
                it.remove();
            }
        }
    }

    /** A host-visible vertex buffer of at least {@code bytes}, valid for this frame only. */
    public RtBuffer acquireVertex(RtContext ctx, long bytes, String label) {
        return acquire(ctx, bytes, VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, label);
    }

    /** A host-visible index buffer of at least {@code bytes}, valid for this frame only. */
    public RtBuffer acquireIndex(RtContext ctx, long bytes, String label) {
        return acquire(ctx, bytes, VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, label);
    }

    private RtBuffer acquire(RtContext ctx, long bytes, int usage, String label) {
        RtBuffer b = ctx.createBuffer(Math.max(bytes, MIN_SIZE), usage, true, label);
        acquiredThisFrame.add(b);
        return b;
    }

    /** Queue everything acquired this frame for destruction once it is safely out of flight. */
    public void endFrame(long frameCounter) {
        for (RtBuffer b : acquiredThisFrame) {
            deferred.add(new Deferred(frameCounter + KEEP_FRAMES, b));
        }
        acquiredThisFrame.clear();
    }

    /** Immediate teardown; only valid once the device is idle (mirrors {@code RtComposite.destroy}). */
    public void destroy() {
        for (RtBuffer b : acquiredThisFrame) {
            b.destroy();
        }
        acquiredThisFrame.clear();
        for (Deferred d : deferred) {
            d.buffer.destroy();
        }
        deferred.clear();
    }
}
