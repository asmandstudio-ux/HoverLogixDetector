package nl.hoverlogix.detect

import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * Review one frame: confirm good boxes, mark wrong ones, fix boundaries or class, draw missed objects.
 * Saving writes the frame + corrected YOLO labels to the corrections dataset.
 */
class EditorActivity : AppCompatActivity(), BoxEditorView.Listener {

    private lateinit var editor: BoxEditorView
    private lateinit var info: TextView
    private lateinit var progress: ProgressBar
    private lateinit var btnWrong: Button
    private lateinit var btnAdd: Button
    private lateinit var btnUndo: Button
    private lateinit var btnReset: Button
    private lateinit var btnSave: Button
    private val classButtons = ArrayList<Button>()

    private var frame: Bitmap? = null
    private var source = "frame"
    private var dirty = false
    private var currentClass = 0
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Labels.load(this)
        setContentView(R.layout.activity_editor)
        editor = findViewById(R.id.editor)
        info = findViewById(R.id.info)
        progress = findViewById(R.id.progress)
        btnWrong = findViewById(R.id.btnWrong)
        btnAdd = findViewById(R.id.btnAdd)
        btnUndo = findViewById(R.id.btnUndo)
        btnReset = findViewById(R.id.btnReset)
        btnSave = findViewById(R.id.btnSave)
        editor.listener = this

        source = intent.getStringExtra(EXTRA_SOURCE) ?: "frame"
        val path = intent.getStringExtra(EXTRA_FRAME_PATH)
        val bmp = path?.let { FrameStore.get(it) }
        if (bmp == null) {
            Toast.makeText(this, "Frame could not be loaded", Toast.LENGTH_LONG).show()
            finish(); return
        }
        frame = bmp
        editor.setImage(bmp)

        buildClassButtons()
        btnWrong.setOnClickListener { editor.toggleRejectSelected() }
        btnAdd.setOnClickListener {
            editor.newBoxClass = currentClass
            editor.addMode = !editor.addMode
            updateButtons()
        }
        btnUndo.setOnClickListener { editor.undo() }
        btnReset.setOnClickListener { editor.resetSelected() }
        btnSave.setOnClickListener { save() }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (!dirty) { finish(); return }
                AlertDialog.Builder(this@EditorActivity)
                    .setTitle("Discard your corrections?")
                    .setPositiveButton("Discard") { _, _ -> finish() }
                    .setNeutralButton("Save") { _, _ -> save() }
                    .setNegativeButton("Keep editing", null)
                    .show()
            }
        })

        // Always review with the accurate model on the full-resolution frame.
        progress.visibility = View.VISIBLE
        info.text = "Detecting…"
        val conf = Settings.confidence(this)
        executor.execute {
            val result = runCatching { Detectors.get(this, Detectors.Kind.ACCURATE).detect(bmp, conf) }
            runOnUiThread {
                progress.visibility = View.GONE
                result.onSuccess { editor.setDetections(it) }
                    .onFailure { info.text = "Detection failed: ${it.message}. You can still draw boxes." }
                dirty = false
                updateButtons()
            }
        }
        updateButtons()
    }

    private fun buildClassButtons() {
        val row = findViewById<LinearLayout>(R.id.classRow)
        Labels.all().forEachIndexed { i, name ->
            val b = Button(this).apply {
                text = name.replace('_', ' ')
                isAllCaps = false
                textSize = 12f
                setTextColor(Color.BLACK)
                backgroundTintList = ColorStateList.valueOf(Labels.color(i))
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginEnd = (4 * resources.displayMetrics.density).toInt()
                }
                setOnClickListener {
                    currentClass = i
                    editor.newBoxClass = i
                    editor.setClassOfSelected(i)   // re-labels the selected box, if any
                    updateButtons()
                }
            }
            classButtons.add(b)
            row.addView(b)
        }
    }

    private fun updateButtons() {
        val sel = editor.selected
        classButtons.forEachIndexed { i, b ->
            val active = if (sel != null) sel.cls == i else currentClass == i
            b.alpha = if (active) 1f else 0.45f
        }
        btnWrong.isEnabled = sel != null
        btnWrong.text = when {
            sel == null -> "Mark wrong"
            !sel.fromModel -> "Delete box"
            sel.rejected -> "Restore"
            else -> "Mark wrong"
        }
        btnReset.isEnabled = sel != null && sel.fromModel && sel.status() != "confirmed"
        btnUndo.isEnabled = editor.canUndo()
        btnAdd.text = if (editor.addMode) "Drag to draw…" else "+ Add box"

        val all = editor.boxes
        val counts = all.groupingBy { it.status() }.eachCount()
        val base = "${all.size} boxes · " + listOf("confirmed", "corrected", "rejected", "added")
            .joinToString(" · ") { "${counts[it] ?: 0} $it" }
        info.text = when {
            editor.addMode -> "Drag on the image to draw a ${Labels.name(currentClass).replace('_', ' ')} box. $base"
            sel != null -> "Selected ${Labels.name(sel.cls)} (${sel.status()}). Drag to move, drag a corner to resize, tap a class to relabel. Pinch to zoom."
            else -> "Tap a box to fix it. Pinch to zoom, two fingers to pan. $base"
        }
    }

    override fun onSelectionChanged(box: EditBox?) {
        box?.let { currentClass = it.cls }
        updateButtons()
    }

    override fun onBoxesChanged() {
        dirty = true
        updateButtons()
    }

    private fun save() {
        val bmp = frame ?: return
        btnSave.isEnabled = false
        val boxes = editor.boxes.map { it.copy() }
        executor.execute {
            val result = runCatching { CorrectionStore.save(this, bmp, boxes, source) }
            runOnUiThread {
                result.onSuccess {
                    Toast.makeText(this, "Saved (${CorrectionStore.count(this)} corrected frames)", Toast.LENGTH_SHORT).show()
                    dirty = false
                    finish()
                }.onFailure {
                    btnSave.isEnabled = true
                    Toast.makeText(this, "Save failed: ${it.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
    }

    companion object {
        const val EXTRA_FRAME_PATH = "frame_path"
        const val EXTRA_SOURCE = "source"
    }
}
