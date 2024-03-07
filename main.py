from typing import Union
from  dotenv  import  load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from npTraitImg import getImageTraitement
from fastapi.responses import JSONResponse
from pass_w import hash_password, check_password    
import numpy as np

app = FastAPI()

items = [
    {"id": 1, "name": "Article 1", "prix": 20.99},
    {"id": 2, "name": "Article 2", "prix": 15.49},
    {"id": 3, "name": "Article 3", "prix": 30.00},
    {"id": 4, "name": "Article 4", "prix": 12.79},
]
def getImageTraitement(Image: str):
    image_path = f"./img/{Image}.jpg"
    # image = Image.open(image_path).convert("L")  # "L" pour convertir en niveaux de gris
    image_matrix = np.array(image_path)
    return image_matrix

@app.get("/image/{img}")
async def getImage(img: str):
    image_matrix = getImageTraitement(img)
    return JSONResponse(content={"image_matrix": image_matrix.tolist()})

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_name": item.price, "item_id": item_id}


@app.get("/items")
async def get_all_items():
    items_response = {"items": [{"id": item["id"], "name": item["name"], "prix": item["prix"]} for item in items]}
    return items_response

@app.get("/image/{img}")
async def getImage(img:str): 
    image_matrix = getImageTraitement(img)
    return JSONResponse(content={"image_matrix": image_matrix.tolist()})

@app.get("/password")
async def get_password():
    password_to_hash = "password"
    hashed_password = hash_password(password_to_hash)
    print(f"Mot de passe haché : {hashed_password}")
    return "password" , hashed_password

@app.get("/password/view")
async def get_view_password(hashed_password: str):
    print(f"Mot de passe haché : {hashed_password}")
    return "password view : ", 