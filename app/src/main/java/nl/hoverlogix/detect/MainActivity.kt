package nl.hoverlogix.detect

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var countText: TextView
    private val io = Executors.newSingleThreadExecutor()

    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            startActivity(Intent(this, VideoActivity::class.java).apply {
                data = uri
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            })
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Labels.load(this)
        setContentView(R.layout.activity_main)
        countText = findViewById(R.id.correctionCount)

        findViewById<Button>(R.id.btnLive).setOnClickListener {
            startActivity(Intent(this, LiveActivity::class.java))
        }
        findViewById<Button>(R.id.btnVideo).setOnClickListener {
            pickVideo.launch(arrayOf("video/*"))
        }
        findViewById<Button>(R.id.btnExport).setOnClickListener { exportCorrections() }
        findViewById<Button>(R.id.btnClear).setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Delete all saved corrections?")
                .setMessage("Export them first if you still need them.")
                .setPositiveButton("Delete") { _, _ -> CorrectionStore.clear(this); refreshCount() }
                .setNegativeButton("Cancel", null)
                .show()
        }

        // Load both models in the background so the first screen opens instantly.
        io.execute {
            runCatching {
                Detectors.get(this, Detectors.Kind.FAST)
                Detectors.get(this, Detectors.Kind.ACCURATE)
            }.onFailure { e ->
                runOnUiThread { Toast.makeText(this, "Model failed to load: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refreshCount()
    }

    private fun refreshCount() {
        val n = CorrectionStore.count(this)
        countText.text = if (n == 0) "No corrected frames saved yet"
        else "$n corrected frame${if (n == 1) "" else "s"} saved, ready to export for retraining"
    }

    private fun exportCorrections() {
        io.execute {
            val uri = CorrectionStore.exportZip(this)
            runOnUiThread {
                if (uri == null) {
                    Toast.makeText(this, "Nothing to export yet", Toast.LENGTH_SHORT).show()
                } else {
                    val send = Intent(Intent.ACTION_SEND).apply {
                        type = "application/zip"
                        putExtra(Intent.EXTRA_STREAM, uri)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(Intent.createChooser(send, "Export corrections"))
                }
            }
        }
    }
}
