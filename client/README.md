
# Dev

### Generate API Interfaces

This method relies on the OpenAPI interface documentation and generator. The specific script is located at `tools/OpenapiToTS.js`. Configure `NEXT_PUBLIC_BACK_API_DOC_HOST` or `NEXT_PUBLIC_BACK_API_HOST` in the `.env` file to point to the OpenAPI doc configuration URL, then run the following command to generate the encapsulated API calls:

```bash
yarn generate:api
```

### Generate Enums

To ensure consistency between frontend and backend enums, generate enum files. The specific script is located at `tools/JEnumToTS.js`. Ensure the backend Java program is placed in the local file directory, then configure `BACKEND_API_ROOT_DIR` in the `.env` file to point to the program directory, and run the following command:

```bash
yarn generate:enum
```

### Generate Contract ABI

To generate and use contract ABI, place the contract ABI files in the contract directory. The specific script is located at `tools/CopyContractABIToTS.js`. Ensure the contract program is placed in the local file directory, then configure `CONTRACT_ROOT_DIR` in the `.env` file to point to the contract directory.

Then, in the contract root directory, run the following command:

```bash
hardhat compile
```

Next, return to the frontend project root directory and run the following command:

```bash
yarn generate:contract:abi
```

