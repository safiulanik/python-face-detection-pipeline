# Steps to run the project

- Run `pip install -r requirements.txt`
- Add image in project root

# SQLs:
CREATE EXTENSION vector;
CREATE TABLE pictures (picture text PRIMARY KEY, embedding vector(768));
