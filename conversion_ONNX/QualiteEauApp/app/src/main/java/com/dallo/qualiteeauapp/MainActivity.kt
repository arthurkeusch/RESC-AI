package com.dallo.qualiteeauapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import ai.onnxruntime.*
import ai.onnxruntime.R
import org.json.JSONObject
import java.nio.charset.Charset

class MainActivity : AppCompatActivity() {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        env = OrtEnvironment.getEnvironment()
        session = loadModel()

        val btnPredict = findViewById<Button>(R.id.btnPredict)
        val txtResult = findViewById<TextView>(R.id.txtResult)

        btnPredict.setOnClickListener {
            val features = loadJsonFeatures()
            val prediction = runInference(features)

            txtResult.text = when (prediction.toInt()) {
                1 -> "Résultat : 1 (Eau de bonne qualité)"
                0 -> "Résultat : 0 (Eau de mauvaise qualité)"
                else -> "Résultat inconnu : $prediction"
            }
        }
    }

    private fun loadModel(): OrtSession {
        val modelBytes = assets.open("modele_qualite_eau.onnx").readBytes()
        return env.createSession(modelBytes)
    }

    private fun loadJsonFeatures(): FloatArray {
        val jsonString = assets.open("input_data.json").readBytes()
            .toString(Charset.defaultCharset())

        val json = JSONObject(jsonString)

        val featureNames = listOf(
            "latitude",
            "longitude",
            "Ammonium",
            "Azote Kjeldahl",
            "Carbone Organique",
            "Chlorophylle a",
            "Chlorures",
            "Conductivité à 25°C",
            "Demande Biochimique en oxygène en 5 jours (D.B.O.5)",
            "Demande Chimique en Oxygène (DCO)",
            "Matières en suspension",
            "Nitrates",
            "Nitrites",
            "Orthophosphates (PO4)",
            "Oxygène dissous",
            "Phosphore total",
            "Phéopigments",
            "Potentiel en Hydrogène (pH)",
            "Taux de saturation en oxygène",
            "Température de l'Eau",
            "Turbidité Formazine Néphélométrique",
            "Oxygène dissous_score",
            "Demande Biochimique en oxygène en 5 jours (D.B.O.5)_score",
            "Nitrates_score",
            "Orthophosphates (PO4)_score",
            "Température de l'Eau_score",
            "Turbidité Formazine Néphélométrique_score",
            "Potentiel en Hydrogène (pH)_score",
            "Conductivité à 25°C_score",
            "WQI",
            "qualite_eau_binaire_lag_1",
            "qualite_eau_binaire_lag_2",
            "qualite_eau_binaire_lag_3",
            "qualite_eau_binaire_lag_7"
        )

        return featureNames.map { key ->
            json.getDouble(key).toFloat()
        }.toFloatArray()
    }

    private fun runInference(features: FloatArray): Float {
        val inputTensor = OnnxTensor.createTensor(
            env,
            arrayOf(features), // float[][]
            longArrayOf(1, features.size.toLong())
        )

        val inputs = mapOf("input" to inputTensor)
        val results = session.run(inputs)

        val output = results[0].value as Array<FloatArray>
        return output[0][0]
    }
}
