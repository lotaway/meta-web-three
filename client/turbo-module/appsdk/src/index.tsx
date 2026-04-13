import Appsdk from './NativeAppsdk'

export default Appsdk

export const generateChallenge = (): string => {
    const bytes = new Uint8Array(32)
    for (let i = 0; i < 32; i++) {
        bytes[i] = Math.floor(Math.random() * 256)
    }
    return Array.from(bytes)
        .map(b => b.toString(16).padStart(2, '0'))
        .join('')
};

