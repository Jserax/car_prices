import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
import pandas as pd


class CarFeatures(BaseModel):
    brand: str
    name: str
    bodyType: str
    color: str
    fuelType: str
    year: float
    mileage: float
    transmission: str
    power: float
    engineDisplacement: str
    location: str


input_spec = JSON(pydantic_model=CarFeatures)
runner = bentoml.mlflow.get("car_price:latest").to_runner()
svc = bentoml.Service('car_price', runners=[runner])


@svc.api(input=input_spec, output=NumpyNdarray(), route='/predict')
def predict(input_data: CarFeatures) -> None:
    df = pd.DataFrame([input_data.dict()])
    df['engineDisplacement'] = df['engineDisplacement'].str.split(' ') \
        .str[0].astype('float32')
    return runner.run(df)
