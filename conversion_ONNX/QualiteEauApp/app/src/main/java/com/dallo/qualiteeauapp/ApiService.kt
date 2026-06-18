package com.example.qualiteeauapp

import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

data class PredictionResponse(val prediction: Int)

interface ApiService {
    @POST("predict")
    fun predict(@Body body: Map<String, Any>): Call<PredictionResponse>
}
