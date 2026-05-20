import type { StyleProp, ViewStyle } from 'react-native';

export type OnScanSuccessEventPayload = {
  data: string;
};

export type OnErrorPayload = {
  message: string;
};

export type ScannerModuleEvents = {
  onScanSuccess: (params: OnScanSuccessEventPayload) => void;
  onError: (params: OnErrorPayload) => void;
};

export type ScannerModuleViewProps = {
  isScanning?: boolean;
  style?: StyleProp<ViewStyle>;
  onScanSuccess?: (event: { nativeEvent: OnScanSuccessEventPayload }) => void;
  onError?: (event: { nativeEvent: OnErrorPayload }) => void;
};
