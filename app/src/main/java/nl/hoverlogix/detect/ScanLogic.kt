package nl.hoverlogix.detect

import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** A location label that may belong to the scanned pallet. Box in pixels of the frame it was found in. */
data class LocationCandidate(val box: RectF, val belongs: Boolean, val score: Float)

/**
 * Geometry for the guided scan: which location labels belong to a pallet, and how far the camera
 * can zoom on a target without pushing it out of the picture.
 */
object ScanLogic {
    const val PALLET = 0
    const val PALLET_LABEL = 1
    const val LOCATION_LABEL = 2
    const val MAX_CANDIDATES = 4

    /** The pallet label closest to the middle of the frame. */
    fun centreMost(labels: List<Detection>, w: Int, h: Int): Detection? =
        labels.minByOrNull { abs(it.box.centerX() - w / 2f) / w + abs(it.box.centerY() - h / 2f) / h }

    /**
     * Ranks the location labels for the pallet that carries [palletLabel].
     *
     * Location labels sit on the beams. A pallet's own label is on the beam it stands on, or on the
     * beam above it (labels with a down arrow). So a label "belongs" when it lies within the pallet's
     * width and close to the pallet's bottom or top edge. Pallets stacked on each other count as one
     * column. Labels that do not meet the rule are still returned, lower in the list, so the user can
     * always pick by hand.
     */
    fun locationCandidates(palletLabel: Detection, dets: List<Detection>): List<LocationCandidate> {
        val locs = dets.filter { it.cls == LOCATION_LABEL }
        if (locs.isEmpty()) return emptyList()
        val column = palletColumn(palletLabel, dets.filter { it.cls == PALLET })
        val w = max(1f, column.width())
        val h = max(1f, column.height())
        return locs.map { l ->
            val cx = l.box.centerX()
            val lo = column.left - 0.15f * w
            val hi = column.right + 0.15f * w
            val hx = when {
                cx < lo -> (lo - cx) / w
                cx > hi -> (cx - hi) / w
                else -> 0f
            }
            val below = l.box.top - column.bottom      // label under the pallet (beam it stands on)
            val above = column.top - l.box.bottom      // label over the pallet (beam above)
            val vGap = when {
                below >= -0.15f * h -> max(0f, below) / h
                above >= -0.15f * h -> max(0f, above) / h
                else -> 1f                              // label in the middle of the load: not a beam label
            }
            val belongs = hx == 0f && vGap <= 0.5f
            val score = hx * 2f + vGap + 0.3f * abs(cx - column.centerX()) / w
            LocationCandidate(RectF(l.box), belongs, score)
        }.sortedWith(compareBy<LocationCandidate> { !it.belongs }.thenBy { it.score }).take(MAX_CANDIDATES)
    }

    /** The pallet holding the label, extended over pallets stacked directly under or on top of it. */
    private fun palletColumn(label: Detection, pallets: List<Detection>): RectF {
        val cx = label.box.centerX()
        val cy = label.box.centerY()
        val own = pallets.filter { it.box.contains(cx, cy) }.minByOrNull { it.box.width() * it.box.height() }
        if (own == null) {
            // No pallet box: assume a load about three labels wide around the label.
            val lw = label.box.width()
            return RectF(label.box.left - lw, label.box.top, label.box.right + lw, label.box.bottom)
        }
        val col = RectF(own.box)
        var grown = true
        while (grown) {
            grown = false
            for (p in pallets) {
                val b = p.box
                if (b.left >= col.left && b.right <= col.right && b.top >= col.top && b.bottom <= col.bottom) continue
                val overlap = min(b.right, col.right) - max(b.left, col.left)
                if (overlap < 0.5f * min(b.width(), col.width())) continue
                val tol = 0.2f * min(b.height(), col.height())
                val touchesBelow = abs(b.top - col.bottom) <= tol
                val touchesAbove = abs(b.bottom - col.top) <= tol
                if (touchesBelow || touchesAbove) { col.union(b); grown = true }
            }
        }
        return col
    }

    fun normalise(box: RectF, w: Int, h: Int) = RectF(box.left / w, box.top / h, box.right / w, box.bottom / h)

    fun union(boxes: List<RectF>): RectF {
        val r = RectF(boxes.first())
        boxes.drop(1).forEach { r.union(it) }
        return r
    }

    /**
     * Largest zoom that keeps [target] (normalised 0..1, at zoom 1) inside the picture.
     * A phone zooms around the middle of the frame, so an off-centre target allows less zoom.
     */
    fun zoomFor(target: RectF, maxZoom: Float): Float {
        val reach = max(
            max(abs(target.left - 0.5f), abs(target.right - 0.5f)),
            max(abs(target.top - 0.5f), abs(target.bottom - 0.5f))
        )
        val keepInFrame = 0.44f / max(reach, 0.01f)
        val fill = 0.6f / max(max(target.width(), target.height()), 0.01f)
        return min(min(keepInFrame, fill), maxZoom).coerceAtLeast(1f)
    }

    /** Where a normalised box from the zoom-1 frame lands after zooming [zoom] times around the centre. */
    fun zoomRect(r: RectF, zoom: Float) = RectF(
        0.5f + (r.left - 0.5f) * zoom, 0.5f + (r.top - 0.5f) * zoom,
        0.5f + (r.right - 0.5f) * zoom, 0.5f + (r.bottom - 0.5f) * zoom
    )

    /**
     * Finds the detection of class [cls] in the zoomed still that matches the expected place
     * (hands move between frames). Falls back to the expected box, enlarged, when none is near.
     * Returns pixels of the still.
     */
    fun relocate(expected: RectF, dets: List<Detection>, cls: Int, w: Int, h: Int, taken: MutableSet<Detection>): RectF {
        val ex = RectF(expected.left * w, expected.top * h, expected.right * w, expected.bottom * h)
        val reach = 0.6f * max(ex.width(), ex.height()) + 0.08f * max(w, h)
        val best = dets.filter { it.cls == cls && it !in taken }
            .map { it to (abs(it.box.centerX() - ex.centerX()) + abs(it.box.centerY() - ex.centerY())) }
            .filter { it.second <= reach }
            .minByOrNull { it.second }?.first
        if (best != null) { taken.add(best); return RectF(best.box) }
        val gx = ex.width() * 0.3f
        val gy = ex.height() * 0.3f
        return RectF(ex.left - gx, ex.top - gy, ex.right + gx, ex.bottom + gy)
    }

    /** Cuts [box] plus a margin out of [src]; small crops are enlarged so barcodes have enough pixels. */
    fun crop(src: Bitmap, box: RectF, margin: Float, minWidth: Int): Bitmap? {
        val mx = box.width() * margin
        val my = box.height() * margin
        val l = (box.left - mx).toInt().coerceIn(0, src.width - 1)
        val t = (box.top - my).toInt().coerceIn(0, src.height - 1)
        val r = (box.right + mx).toInt().coerceIn(l + 1, src.width)
        val b = (box.bottom + my).toInt().coerceIn(t + 1, src.height)
        if (r - l < 8 || b - t < 8) return null
        val cut = Bitmap.createBitmap(src, l, t, r - l, b - t)
        if (cut.width >= minWidth) return cut
        val s = minWidth.toFloat() / cut.width
        return Bitmap.createScaledBitmap(cut, minWidth, (cut.height * s).toInt().coerceAtLeast(1), true)
    }
}
