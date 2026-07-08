
Build the website:

```
./import_csv_results.py
./render_website.sh
```

Publish with GitHub Pages:

1. Push to `main`.
2. In GitHub repo settings, open `Pages` and set `Source` to `GitHub Actions`.
3. Workflow in `.github/workflows/deploy-pages.yml` will build `website/_site` and deploy it.

View it:

```
docker run --rm -p 8080:80 -v "${PWD}/website/_site:/usr/share/nginx/html:ro" nginx
```
