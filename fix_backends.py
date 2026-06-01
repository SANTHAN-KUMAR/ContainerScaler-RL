import yaml

with open('deploy/online-boutique.yaml', 'r') as f:
    docs = list(yaml.safe_load_all(f))

for doc in docs:
    if doc and doc.get('kind') == 'Deployment':
        name = doc['metadata']['name']
        if name != 'frontend':
            for container in doc['spec']['template']['spec']['containers']:
                if 'resources' in container and 'limits' in container['resources']:
                    if 'cpu' in container['resources']['limits']:
                        container['resources']['limits']['cpu'] = '1000m'

with open('deploy/online-boutique.yaml', 'w') as f:
    yaml.safe_dump_all(docs, f, sort_keys=False)
