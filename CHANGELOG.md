# Changelog

## [1.2.0](https://github.com/Emmi-AI/noether/compare/v1.1.0...v1.2.0) (2026-03-03)


### ✨ Features

* add parameter validity check to models ([#90](https://github.com/Emmi-AI/noether/issues/90)) ([7e619f2](https://github.com/Emmi-AI/noether/commit/7e619f206ffb1803ec3f010c6f358ca3e6a14a49))
* add shared field propagation from parent to submodules for UPT schema ([#80](https://github.com/Emmi-AI/noether/issues/80)) ([6289105](https://github.com/Emmi-AI/noether/commit/628910599c0683420b61a30480024de4ffde920c))
* Track step timings mean and max on exp tracker ([#82](https://github.com/Emmi-AI/noether/issues/82)) ([add8975](https://github.com/Emmi-AI/noether/commit/add8975a32a829c8c556d1aa281b4e3a41e57992))


### 🐛 Bug Fixes

* adjusted failing test ([#95](https://github.com/Emmi-AI/noether/issues/95)) ([66a336c](https://github.com/Emmi-AI/noether/commit/66a336ce8448ced6bf541269a03cb51d8893be27))
* autograd error when skip_nan_loss is enabled ([#86](https://github.com/Emmi-AI/noether/issues/86)) ([1ff9873](https://github.com/Emmi-AI/noether/commit/1ff9873ccaf8e598695700893b4fb6aca0c55188))
* docstrings for UPT args ([#91](https://github.com/Emmi-AI/noether/issues/91)) ([ee47116](https://github.com/Emmi-AI/noether/commit/ee47116a30b0a49777242c3953cdd5ffd2efcb08))
* error when no callback samplers are set ([#78](https://github.com/Emmi-AI/noether/issues/78)) ([0beff9e](https://github.com/Emmi-AI/noether/commit/0beff9e179eadc239e336a69f02ec2750182c310))
* Fix recursive field injection ([#84](https://github.com/Emmi-AI/noether/issues/84)) ([7973869](https://github.com/Emmi-AI/noether/commit/7973869f0d30bb537a676bb9862fe3de72f9924c))
* ignore name config from name extraction ([#71](https://github.com/Emmi-AI/noether/issues/71)) ([8e6a602](https://github.com/Emmi-AI/noether/commit/8e6a60286797601e754c5359996779dd52f8b01a))
* missing stingify and re-export ([#68](https://github.com/Emmi-AI/noether/issues/68)) ([607ed14](https://github.com/Emmi-AI/noether/commit/607ed14d6626c510520bd286d523f47a70a32cfb))
* Never link a job to the resumption of itself ([#61](https://github.com/Emmi-AI/noether/issues/61)) ([108122b](https://github.com/Emmi-AI/noether/commit/108122b4c87dc4fb5c4df8a392eb6fb181c1a132))
* persist order from dictionaries when saving out job config ([#79](https://github.com/Emmi-AI/noether/issues/79)) ([1158b51](https://github.com/Emmi-AI/noether/commit/1158b51ed9c529f43cacf798dc7dce8d50b9514b))
* Tracker filenaming to avoid circular import when dependencies don't exist ([#77](https://github.com/Emmi-AI/noether/issues/77)) ([713e0bb](https://github.com/Emmi-AI/noether/commit/713e0bbca9d15aba95bb4c59b99e7ad5bec3399f))
* typing issues in data module ([#69](https://github.com/Emmi-AI/noether/issues/69)) ([764cc95](https://github.com/Emmi-AI/noether/commit/764cc955ea4f3d0e870628a217bc57b5d448dd40))
* update install instructions in the tutorial README ([#75](https://github.com/Emmi-AI/noether/issues/75)) ([62205b0](https://github.com/Emmi-AI/noether/commit/62205b0056c329bbfa765d18e321d8d3f984c6e2))


### ⚡ Performance Improvements

* Avoid device synchronize in supernode pooling ([#93](https://github.com/Emmi-AI/noether/issues/93)) ([f82850f](https://github.com/Emmi-AI/noether/commit/f82850fcdcf03afc3b67d2e70816112f4172c1a0))


### ♻️ Code Refactoring

* refactor and fix failing checkpoint loading inference module ([#66](https://github.com/Emmi-AI/noether/issues/66)) ([c03bc80](https://github.com/Emmi-AI/noether/commit/c03bc8096490ad1958fb3827c6ee36b772ec69be))
* refactored the initializers and checkpointing  ([#59](https://github.com/Emmi-AI/noether/issues/59)) ([d7893bc](https://github.com/Emmi-AI/noether/commit/d7893bc83ec588f67e0c70b6f72f5371df71032d))

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
