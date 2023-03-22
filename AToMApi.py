from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import uvicorn
import base64
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/linearregression/")
async def linear_regression(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents), delimiter=',')
    X = df[['x']]
    y = df['y']
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    plt.scatter(df['x'], df['y'])
    plt.plot(df['x'], model.predict(X), color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "image": img_base64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)