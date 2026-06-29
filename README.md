
Build the website:

```
./import_csv_results.py
./render_website.sh
```

View it:

```
docker run --rm -p 8080:80 -v "${PWD}/website/_site:/usr/share/nginx/html:ro" nginx
```
