// Reexport the native module. On web, it will be resolved to ScannerModule.web.ts
// and on native platforms to ScannerModule.ts
export { default } from './src/ScannerModule';
export { default as ScannerView } from './src/ScannerView';
export * from  './src/Scanner.types';
