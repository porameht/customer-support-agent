from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from mongodb import client, db, collection, ATLAS_VECTOR_SEARCH_INDEX_NAME

load_dotenv()

class Package(BaseModel):
    text: str
    name: str
    price: float
    currency: str
    facebook_pages: int
    line_connections: str
    admin_support: str

def create_package_summary(package: Package) -> str:
    return f"Package {package.name} - ฿{package.price}/month. Facebook Pages: {package.facebook_pages}, {package.line_connections}, {package.admin_support}"

async def seed_database():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")

        collection.delete_many({})
        
        packages = [
            Package(
                name="S",
                price=990,
                currency="฿",
                facebook_pages=5,
                line_connections="เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด",
                admin_support="แอดมินดูแล 24 ชม.",
                text="Package S - ฿990/เดือน\n- เชื่อมต่อ Facebook Page: 5 Pages\n- เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด\n- แอดมินดูแล 24 ชม.",  # Initialize empty string
            ),
            Package(
                name="M", 
                price=1900,
                currency="฿",
                facebook_pages=10,
                line_connections="เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด",
                admin_support="แอดมินดูแล 24 ชม.",
                text="Package M - ฿1,900/เดือน\n- เชื่อมต่อ Facebook Page: 10 Pages\n- เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด\n- แอดมินดูแล 24 ชม.",
            ),
            Package(
                name="L",
                price=4900,
                currency="฿", 
                facebook_pages=20,
                line_connections="เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด",
                admin_support="แอดมินดูแล 24 ชม.",
                text="Package L - ฿4,900/เดือน\n- เชื่อมต่อ Facebook Page: 20 Pages\n- เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด\n- แอดมินดูแล 24 ชม.",
            ),
            Package(
                name="XL",
                price=12500,
                currency="฿",
                facebook_pages=30,
                line_connections="เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด", 
                admin_support="แอดมินดูแล 24 ชม.",
                text="Package XL - ฿12,500/เดือน\n- เชื่อมต่อ Facebook Page: 30 Pages\n- เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด\n- แอดมินดูแล 24 ชม.",
            ),
            Package(
                name="4XL",
                price=25000,
                currency="฿",
                facebook_pages=50,
                line_connections="เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด",
                admin_support="แอดมินดูแล 24 ชม.",
                text="Package 4XL - ฿25,000/เดือน\n- เชื่อมต่อ Facebook Page: 50 Pages\n- เชื่อมต่อ Line Official/Line My shop ได้ไม่จำกัด\n- แอดมินดูแล 24 ชม.",
            )
        ]

        embedding_model = OpenAIEmbeddings()

        for package in packages:
            # Create the summary and set it as embedding_text
            package.text = create_package_summary(package)
            
            # Create document for MongoDB with page_content field
            document = {
                "embedding_text": package.text,
                "metadata": package.model_dump()
            }
            
            # Generate embedding
            embedding = embedding_model.embed_documents([document["embedding_text"]])[0]
            document["embedding"] = embedding

            # Insert directly into MongoDB
            collection.insert_one(document)

            print(f"Successfully processed & saved package: {package.name}")

        print("Database seeding completed")

    except Exception as error:
        print("Error seeding database:", error)
    finally:
        client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(seed_database())