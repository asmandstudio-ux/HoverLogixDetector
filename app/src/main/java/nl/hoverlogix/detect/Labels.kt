package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Color

/** Class names (from assets/labels.txt) and one colour per class. */
object Labels {
    private var names: List<String> = listOf("pallet", "pallet_label", "location_label", "beam", "upright", "empty_location", "pallet_base")

    // Same colours as the annotation previews: blue, pink, yellow, red-orange, green, cyan, brown.
    private val colors = intArrayOf(
        Color.rgb(0, 120, 255),
        Color.rgb(255, 0, 255),
        Color.rgb(255, 220, 0),
        Color.rgb(255, 60, 0),
        Color.rgb(0, 230, 120),
        Color.rgb(90, 216, 255),
        Color.rgb(200, 120, 60)
    )

    fun load(context: Context) {
        runCatching {
            context.assets.open("labels.txt").bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotEmpty() }
        }.getOrNull()?.takeIf { it.isNotEmpty() }?.let { names = it }
    }

    val count: Int get() = names.size
    fun name(cls: Int): String = names.getOrElse(cls) { "class$cls" }
    fun all(): List<String> = names
    fun color(cls: Int): Int = colors[((cls % colors.size) + colors.size) % colors.size]
}

/** Small persisted settings shared by the screens. */
object Settings {
    private const val PREFS = "hoverlogix"

    fun confidence(context: Context): Float =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat("conf", 0.35f)

    fun setConfidence(context: Context, v: Float) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putFloat("conf", v).apply()

    fun liveModel(context: Context): Detectors.Kind =
        if (context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("liveAccurate", true))
            Detectors.Kind.ACCURATE else Detectors.Kind.FAST

    fun setLiveModel(context: Context, kind: Detectors.Kind) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putBoolean("liveAccurate", kind == Detectors.Kind.ACCURATE).apply()

    fun videoModel(context: Context): Detectors.Kind =
        if (context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("videoFast", false))
            Detectors.Kind.FAST else Detectors.Kind.ACCURATE

    fun setVideoModel(context: Context, kind: Detectors.Kind) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putBoolean("videoFast", kind == Detectors.Kind.FAST).apply()
}
