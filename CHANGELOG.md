# Changelog

## [2.0.0](https://github.com/Emmi-AI/noether/compare/v1.1.0...v2.0.0) (2026-03-27)


### ⚠ BREAKING CHANGES

* Add kv caching to AB-UPT ([#131](https://github.com/Emmi-AI/noether/issues/131))
* simplify model init methods by moving config class instantiation to the config classes themself ([#110](https://github.com/Emmi-AI/noether/issues/110))

### ✨ Features

* add attention mask support for padded token sequences ([#115](https://github.com/Emmi-AI/noether/issues/115)) ([5474841](https://github.com/Emmi-AI/noether/commit/54748411c090f05002b0b47a12fe1f454c23bdaf))
* add cli tool to validate config first and lauch a slurm job via… ([#94](https://github.com/Emmi-AI/noether/issues/94)) ([d734b6a](https://github.com/Emmi-AI/noether/commit/d734b6a3799c2ca9a5042b0dd31ca61b06255176))
* Add kv caching to AB-UPT ([#131](https://github.com/Emmi-AI/noether/issues/131)) ([bfea608](https://github.com/Emmi-AI/noether/commit/bfea608ec5860b6e4358fe41496bd435a4dfd422))
* add noether-init for scaffolding ([#113](https://github.com/Emmi-AI/noether/issues/113)) ([de3b6b9](https://github.com/Emmi-AI/noether/commit/de3b6b93c32bcc92b72dab3e7b101bbcd6f03e35))
* add signal handlers to BaseTrainer ([#111](https://github.com/Emmi-AI/noether/issues/111)) ([5769509](https://github.com/Emmi-AI/noether/commit/5769509dc719b887001ce3c9cf50916adea15aa1))
* add sphinx doctest ([#100](https://github.com/Emmi-AI/noether/issues/100)) ([e3ef887](https://github.com/Emmi-AI/noether/commit/e3ef8873c7b975ca66c8619e0e65a565187b99ba))
* add`noether-init` cli command to create a new boilerplate project to get started with implementing custom models and datasets. After running this command, the train command is printed, which the user can run to start a test training run immediately. ([de3b6b9](https://github.com/Emmi-AI/noether/commit/de3b6b93c32bcc92b72dab3e7b101bbcd6f03e35))
* eliminate requirement for custom root schema ([#108](https://github.com/Emmi-AI/noether/issues/108)) ([81a62ae](https://github.com/Emmi-AI/noether/commit/81a62aea5e512b11dd3e0a7f56b16d5c76c3a9fd))
* match callback state dicts by key instead of position on resume  ([#114](https://github.com/Emmi-AI/noether/issues/114)) ([12bbaba](https://github.com/Emmi-AI/noether/commit/12bbabad8210643423113f1f17927b64c99f7711))
* support for resuming training without data loading ([#130](https://github.com/Emmi-AI/noether/issues/130)) ([492f041](https://github.com/Emmi-AI/noether/commit/492f0417594b2ea5810e0eb12094e8432a1b0b1b)), closes [#55](https://github.com/Emmi-AI/noether/issues/55)


### 🐛 Bug Fixes

* bring back missing tutorial image ([#112](https://github.com/Emmi-AI/noether/issues/112)) ([4772538](https://github.com/Emmi-AI/noether/commit/4772538cb38c61907cd543ff6e200798fea5217a))
* Remove bad lr scheduler default value and make it required instead ([#129](https://github.com/Emmi-AI/noether/issues/129)) ([d4dbb0a](https://github.com/Emmi-AI/noether/commit/d4dbb0a068fd434abf854c59296eefa4eb69e70d))
* remove static config from composite transformer block ([#102](https://github.com/Emmi-AI/noether/issues/102)) ([7b14274](https://github.com/Emmi-AI/noether/commit/7b142741518717dd5d4f1ec654f372f47dd8395e))
* replace wrong dataset class for stats cli tool ([#103](https://github.com/Emmi-AI/noether/issues/103)) ([4cad333](https://github.com/Emmi-AI/noether/commit/4cad333ca04163f0483f064fa8ebe83b2083f333))
* split input sequence correctly and enable using kv_dim ([#116](https://github.com/Emmi-AI/noether/issues/116)) ([77dc41d](https://github.com/Emmi-AI/noether/commit/77dc41d8a8d5e470c693da34db0b75c913b4387d))
* submit job CLI was not intercepting Hydra properly ([#127](https://github.com/Emmi-AI/noether/issues/127)) ([9e624b8](https://github.com/Emmi-AI/noether/commit/9e624b8707a00c26e474f65fbee174a74fc956c8))
* variable duplication bug in caeml preprocessing script to st… ([#125](https://github.com/Emmi-AI/noether/issues/125)) ([22ac5f0](https://github.com/Emmi-AI/noether/commit/22ac5f0d7001730072fd7bf63c9d1358bac62662))
* various smaller bugs ([#104](https://github.com/Emmi-AI/noether/issues/104)) ([308607a](https://github.com/Emmi-AI/noether/commit/308607a8bee6477a4067d5561ac3b63c4ec651b7))
* wrong semantics for BestCheckpoint tolerance config ([#105](https://github.com/Emmi-AI/noether/issues/105)) ([b5895cb](https://github.com/Emmi-AI/noether/commit/b5895cb451a969ca8da6318e4f7c4694dc7678c4))
* zero division in ProgressCallback in edge case ([#122](https://github.com/Emmi-AI/noether/issues/122)) ([b3811b0](https://github.com/Emmi-AI/noether/commit/b3811b0077ceefd71cdebfc346a6671a63350128))


### ♻️ Code Refactoring

* simplify model init methods by moving config class instantiation to the config classes themself ([#110](https://github.com/Emmi-AI/noether/issues/110)) ([ae2ed95](https://github.com/Emmi-AI/noether/commit/ae2ed95842f46c0cc4f7c989cb1775709d71c254))

## [1.1.0](https://github.com/Emmi-AI/noether/compare/v1.0.0...v1.1.0) (2026-02-10)


### ✨ Features

* Add support for trackio ([#45](https://github.com/Emmi-AI/noether/issues/45)) ([37bea81](https://github.com/Emmi-AI/noether/commit/37bea81c3d2d84b51628d23ad85e7e8b1badab53))
* Make torch-cluster dependency optional ([#36](https://github.com/Emmi-AI/noether/issues/36)) ([3a4a8c2](https://github.com/Emmi-AI/noether/commit/3a4a8c2b5d336c8cecd64ec21042193dd9699e9c))
* remove torch-scatter dependency ([#26](https://github.com/Emmi-AI/noether/issues/26)) ([cb57a04](https://github.com/Emmi-AI/noether/commit/cb57a046362f9e0cb6c966d8265c2cf006c395b9))


### 🐛 Bug Fixes

* add help to noether-train and use for installation verification ([#34](https://github.com/Emmi-AI/noether/issues/34)) ([717658c](https://github.com/Emmi-AI/noether/commit/717658cc0f85a961d3e889f4e3c0d1a5d8b06cf4))
* fix failing test, remove unused classes, update wrong link in docs, and update tutorial readme ([#32](https://github.com/Emmi-AI/noether/issues/32)) ([2a4b0b4](https://github.com/Emmi-AI/noether/commit/2a4b0b49d9c8ad5b38c8a4672ca03cce0a1a2164))
* use mypy and pytest config as intended in pyproject.toml ([#40](https://github.com/Emmi-AI/noether/issues/40)) ([165770c](https://github.com/Emmi-AI/noether/commit/165770c46479db6ffdfe5f93e2a44251730090f2))


### 📚 Documentation

* copy to clipboard button ([#46](https://github.com/Emmi-AI/noether/issues/46)) ([bbb7853](https://github.com/Emmi-AI/noether/commit/bbb78530f5bdc855f340719d330809423ae65ba1))
* update callback docs ([#39](https://github.com/Emmi-AI/noether/issues/39)) ([4ea7b4d](https://github.com/Emmi-AI/noether/commit/4ea7b4d43ed2362950a83af7ef33db330f824916))
* update image links on readme for pypi page ([#30](https://github.com/Emmi-AI/noether/issues/30)) ([3243652](https://github.com/Emmi-AI/noether/commit/3243652c86a7d1af2f45dffbb9d5e36fc4dc48a3))
