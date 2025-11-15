# Dataset

## ShapeNet Chairs (Category 03001627)

### Option 1: Use Pre-rendered Images (Recommended)
Download from: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

Extract to:
```
data/
└── aligned/
    ├── model_id_1/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ... 024.png
    └── model_id_2/
        └── ... 024.png
```

### Option 2: Render Yourself
- Clone: https://github.com/vencia/multiview-renderer
- Download ShapeNet Core v2 from https://shapenet.org/
- Run render pipeline (see multiview-renderer README)
- Configure: 24 views, 256×256, grayscale

### Size
- 1,300 chairs × 24 views × ~5MB = ~156GB total
- For quick testing: use 50-100 models
