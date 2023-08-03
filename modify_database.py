from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import confirm
import chromadb
import os
import shutil
import time
from chromadb.config import Settings
from constants import CHROMADB_PERSIST_DIR, CHROMADB_IMPL
import hashlib
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
chroma_client = chromadb.Client(
    Settings(persist_directory=CHROMADB_PERSIST_DIR, chroma_db_impl=CHROMADB_IMPL))

# Initialize prompt_toolkit session
session = PromptSession()


def main():
    while True:
        print("\nWelcome to the PETGPT ChromaDB Management Program!")
        print("\nPlease select an option:")
        print("1. View Collections")
        print("2. Create New Collection")
        print("3. Delete Collection")
        print("4. View Items in a Collection")
        print("5. Add Items to a Collection")
        print("6. Update Items in a Collection")
        print("7. Delete Items from a Collection")
        print("8. Search in a Collection")
        print("9. Search All Collections")
        print("10. Backup Database")
        print("11. Restore Database from Backup")
        print("12. Reset Database")
        print("13. Exit")

        choice = session.prompt("\nEnter your choice: ")

        if choice == "1":
            view_collections()
        elif choice == "2":
            create_collection()
        elif choice == "3":
            delete_collection()
        elif choice == "4":
            view_items()
        elif choice == "5":
            add_items()
        elif choice == "6":
            update_items()
        elif choice == "7":
            delete_items()
        elif choice == "8":
            search_collection()
        elif choice == "9":
            search_all_collections()
        elif choice == "10":
            backup_database()
        elif choice == "11":
            restore_database()
        elif choice == "12":
            reset_database()
        elif choice == "13":
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 13.")


def view_collections():
    print("\nCollections:")
    for i, collection in enumerate(chroma_client.list_collections(), start=1):
        print(f"{i}. {collection}")


def create_collection():
    name = session.prompt("\nEnter a name for the new collection: ")
    chroma_client.get_or_create_collection(name)
    print(f"\nCollection '{name}' created.")


def delete_collection():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to delete: ")) - 1
    collections = chroma_client.list_collections()
    # Extract the name from the Collection object
    collection_name = collections[index].name
    if confirm(f"\nAre you sure you want to delete the collection '{collection_name}'?"):
        chroma_client.delete_collection(collection_name)
        print(f"\nCollection '{collection_name}' deleted.")


def view_items():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to view: ")) - 1
    collections = chroma_client.list_collections()
    # Extract the name from the Collection object
    collection_name = collections[index].name
    collection = chroma_client.get_collection(collection_name)
    items = collection.peek()
    print("\nItems:")
    for i in range(len(items['ids'])):
        print(f"\nItem {i+1}:")
        print(f"ID: {items['ids'][i]}")
        print(f"Document: {items['documents'][i]}")
        print(f"Metadata: {items['metadatas'][i]}")
        print(f"Embedding Length: {len(items['embeddings'][i])}")
        if session.prompt("\nPress Enter to continue to the next item or 'q' to quit: ") == 'q':
            break


def add_items():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to add items to: ")) - 1
    collections = chroma_client.list_collections()
    # Extract the name from the Collection object
    collection_name = collections[index].name
    collection = chroma_client.get_or_create_collection(collection_name)
    document = session.prompt(
        "\nEnter the document to add (press Escape, Enter to submit):\n", multiline=True)
    # Generate a hash of the document and use the first 8 characters as the ID
    document_id = hashlib.sha256(document.encode()).hexdigest()[:8]
    collection.add(documents=[document], ids=[document_id])
    chroma_client.persist()
    print("\nItem added.")


def update_items():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to update items in: ")) - 1
    collections = chroma_client.list_collections()
    collection = collections[index]  # Directly assign the Collection object
    ids = session.prompt(
        "\nEnter the IDs of the items to update (separated by commas): ").split(",")
    documents = session.prompt(
        "\nEnter the new documents for the items (separated by commas): ").split(",")
    collection.upsert(ids=ids, documents=documents)
    chroma_client.persist()
    print("\nItems updated.")


def delete_items():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to delete items from: ")) - 1
    collections = chroma_client.list_collections()
    collection = collections[index]  # Directly assign the Collection object
    ids = session.prompt(
        "\nEnter the IDs of the items to delete (separated by commas): ").split(",")
    if confirm(f"\nAre you sure you want to delete the items with IDs {ids}?"):
        collection.delete(ids=ids)
        chroma_client.persist()
        print("\nItems deleted.")


def search_collection():
    view_collections()
    index = int(session.prompt(
        "\nEnter the number of the collection to search in: ")) - 1
    collections = chroma_client.list_collections()
    # Extract the name from the Collection object
    collection_name = collections[index].name
    collection = chroma_client.get_collection(collection_name)
    query = session.prompt("\nEnter the query: ")
    results = collection.query(query_texts=[query], n_results=2, include=[
                               "documents", "metadatas", "distances"])
    print("\nSearch Results:")
    for i in range(len(results['ids'])):
        print(f"\nResult {i+1}:")
        print(f"ID: {results['ids'][i]}")
        print(f"Document:")
        # Join the list of strings with '\n'
        print('\n'.join(results['documents'][i]))
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Distance: {results['distances'][i]}")


def search_all_collections():
    query = session.prompt("\nEnter the query: ")
    print("\nSearch Results:")
    for collection in chroma_client.list_collections():
        # Extract the name from the Collection object
        collection_name = collection.name
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=2, include=[
                                   "documents", "metadatas", "distances"])
        print(f"\nCollection '{collection_name}':")
        for i in range(len(results['ids'])):
            print(f"\nResult {i+1}:")
            print(f"ID: {results['ids'][i]}")
            print(f"Document:")
            # Join the list of strings with '\n'
            print('\n'.join(results['documents'][i]))
            print(f"Metadata: {results['metadatas'][i]}")
            print(f"Distance: {results['distances'][i]}")


def backup_database():
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)
    backup_name = f"backup_{time.strftime('%Y%m%d%H%M%S')}"
    shutil.copytree("chromadb", os.path.join(backup_dir, backup_name))
    print(f"\nDatabase backed up to '{backup_name}'.")


def restore_database():
    # TODO
    pass


def reset_database():
    if confirm("\nAre you sure you want to reset the database? This action cannot be undone."):
        chroma_client.reset()
        print("\nDatabase reset.")


if __name__ == "__main__":
    main()
