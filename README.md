# Leaderboard for SISAP 2026 Indexing challenge

For challenge details, see <https://sisap-challenges.github.io/2026/index.html>.

## Data source 

We evaluated all the approaches using the TIRA platform at <https://www.tira.io/task-overview/sisap-2026>.
The resulting files are stored at <https://files.webis.de/data-in-progress/data-research/sisap-2025/>.
The script `export_results.py` was used to create the CSV files in `data/`. 

## Local deployment 
Build the website:

```
./import_csv_results.py
./render_website.sh
```

View it:

```
docker run --rm -p 8080:80 -v "${PWD}/website/_site:/usr/share/nginx/html:ro" nginx
```

## Deployment via Github actions
Publish with GitHub Pages:

1. Push to `main`.
2. In GitHub repo settings, open `Pages` and set `Source` to `GitHub Actions`.
3. Workflow in `.github/workflows/deploy-pages.yml` will build `website/_site` and deploy it.

