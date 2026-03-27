from fastapi.templating import Jinja2Templates
try:
    templates = Jinja2Templates(directory="templates")
    print("Templates loaded successfully.")
    for template_name in ["index.html", "predictor.html", "recommender.html", "session.html", "performance.html"]:
        print(f"Loading {template_name}...")
        templates.get_template(template_name)
    print("All templates rendered successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
