# ai-compare-candidates

## Compare and rank multiple candidate objects using artificial intelligence retrieval augmented generation, providing the rationale

This package allows you to rank multiple candidate objects in a customised manner by providing a user-supplied function that converts each such object into a candidate document in the form of a string. It then uses a vector database and embedding to perform a similarity search retrieval of an initial number of candidates. This search is then refined into the same or a smaller number of candidates by feeding them into a large language model which will rank the candidates. The candidate documents or part thereof can also be summarised into a defined word limit.

## Cloning of package

After performing a git clone:

0. This assumes that you have a suitable node version installed. If you don't already have yarn installed globally:

```bash
npm install -g yarn
```

1. Remove the line ```"packageManager": "yarn@4.12.0"``` and the preceding comma in package.json
2. Remove the line ```yarnPath: .yarn/releases/yarn-4.12.0.cjs``` from .yarnrc.yml
3. 

```bash
yarn set version 4.12.0
yarn --immutable
```