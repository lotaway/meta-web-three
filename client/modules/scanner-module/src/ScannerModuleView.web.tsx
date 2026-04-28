import * as React from 'react';

import { ScannerModuleViewProps } from './ScannerModule.types';

export default function ScannerModuleView(props: ScannerModuleViewProps) {
  return (
    <div style={{ width: '100%', height: '100%', background: '#000' }}>
      <p>Scanner is not supported on web</p>
    </div>
  );
}