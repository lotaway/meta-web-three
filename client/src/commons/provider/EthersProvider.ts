import {ethers, BrowserProvider, Eip1193Provider} from "ethers"

export default class EthersProvider {

    static ethersProvider: EthersProvider

    browserProvider: BrowserProvider

    constructor(readonly walletProvider: Eip1193Provider) {
        this.browserProvider = new BrowserProvider(walletProvider)
    }

    static getInstance(walletProvider?: Eip1193Provider): EthersProvider {
        if (!EthersProvider.ethersProvider) {
            if (!walletProvider)
                throw new TypeError(`walletProvider not exist: ${walletProvider}`)
            EthersProvider.ethersProvider = new EthersProvider(walletProvider as Eip1193Provider)
        }
        return EthersProvider.ethersProvider
    }

    async getWalletDefaultAddress() {
        const accounts = await this.browserProvider.send("eth_requestAccounts", [])
        return accounts[0]
    }

    // 读取钱包地址
    async getWalletAddress() {
        return await this.browserProvider.send("eth_requestAccounts", [])
    }

    async getBalance(walletAddress: string) {
        // return await this.browserProvider.send("eth_getBalance", [walletAddress])
        return ethers.formatEther(await this.browserProvider.getBalance(walletAddress))
    }

    async getSigner() {
        return await this.browserProvider.getSigner()
    }

    // 获取钱包签名
    async getWalletSignature(walletAddress: string, timestamp: number = this.getTimestamp(), message?: string) {
        message = message ?? `${walletAddress}|login by wallet|${timestamp}`
        const signer = await this.getSigner()
        return await signer.signMessage(message)
    }

    getTimestamp() {
        return +new Date()
    }

}