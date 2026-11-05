# Datasets and resources

The six recorded evaluation datasets are DET-COMPASS, CLCXray, PIXray, PIDray, DvXray and HiXray. Current full-precision endpoints are in [results/current_six_dataset_AP.csv](../results/current_six_dataset_AP.csv). Dataset role definitions, source fitting and validation/tune partitions belong to the recorded protocols in [reproduction/](../reproduction/README.md), rather than to an inferred folder naming convention.

The source-trained quality heads include CLCXray/PIDray/HiXray supervision; these evaluations are not all unseen-domain transfers. The fixed native F-only transfer comparison has negative outcomes on several datasets; it must not be conflated with the V/F candidate union or final conditional ranking.

External resources include provider images and annotations, DINO/SAM/detector weights, original proposal/mask JSONs, large feature and prediction caches, and the two-environment dependencies used by the raw-image connector. They are identified by the supplied protocols and resource configuration; they are not bundled here. Compact author-fitted downstream quality states are separately identified, not upstream detector weights.

Configure external inputs using the current reproduction resource template. The old `configs/paths.example.env` and [archived July layout guide](legacy/DATASETS_AND_RESOURCES_20260709.md) remain available for the legacy native-stage scripts; they are not a complete environment lock for the current raw-image pipeline.
