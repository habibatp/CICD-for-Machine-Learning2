install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add Model/drug_pipeline.skops
	git add Results/model_results.png
	git add Results/metrics.txt
	# Ne plante pas si rien à commit
	git diff --cached --quiet && echo "No changes to commit" || git commit -m "Update: new model and evaluation results"
	git push --force origin HEAD:update
