import api from "./axios";

export const getOptions = () => api.get("/options")

export const predict = (data) => 
    api.post("/predict-with-similar",data)

export const getHistory = () => 
    api.get("/history")

export const getModelResults = () => 
    api.get("/model-results")

export const getBrandAnalysis = () => 
    api.get("/analysis/brand-price")

export const getReleaseAnalysis = () =>
    api.get("/analysis/release-year-price")

export const getDaysAnalysis = () =>
    api.get("/analysis/days-used-price")
