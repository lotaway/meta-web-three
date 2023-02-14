import React, {createContext, useEffect, useState} from "react";
import {ethers} from "ethers";
import {contractABI, transactionContractAddress} from "../config/constants";

export const TransactionContext = createContext<any>({});

export const {ethereum} = window as any;

const getContacts = () => {
    const provider = new ethers.providers.Web3Provider(ethereum);
    const signer = provider.getSigner();
    const transactionContract = new ethers.Contract(transactionContractAddress, contractABI, signer);
    console.log({
        provider,
        signer,
        transactionContract
    });
    return {transactionContract};
}

export const TransactionProvider = ({children}: { children: React.ReactNode }) => {
    const [isTransacting, setIsTransacting] = useState(false);
    const [currentWalletAccount, setCurrentWalletAccount] = useState("");
    const [transactionCount, setTransactionCount] = useState(0);    //  应使用类似localstorage保存缓存
    const [transactionRecords, setTransactionRecords] = useState([]);
    const initWalletConnect = async () => {
        try {
            if (!ethereum) {
                return alert("Please install metamask");
            }
            const accounts = await ethereum.request({
                method: "eth_accounts"
            });
            accounts.length && setCurrentWalletAccount(accounts[0]);
            void getTransactionRecords();
        } catch (err) {
            console.error(err);
            throw new Error("No ethereum object.");
        }
    };
    const connectWallet = async () => {
        try {
            if (!ethereum) {
                return alert("Please install metamask");
            }
            //  the external account
            const accounts = await ethereum.request({
                method: "eth_requestAccounts"
            });
            setCurrentWalletAccount(accounts[0]);
        } catch (err) {
            console.error(err);
            throw new Error("No ethereum object.");
        }
    };
    const sendTransaction = async ({addressTo, amount, keyword, message}: {
        addressTo: string
        amount: string
        keyword: string
        message: string
    }) => {
        try {
            if (!ethereum) {
                return alert("Please install metamask");
            }
            const {transactionContract} = getContacts();
            //  直接从钱包账户之间转账，没有放入合约账户
            await ethereum.request({
                method: "eth_sendTransaction",
                params: [{
                    from: currentWalletAccount,
                    to: addressTo,
                    gas: "0x5208",  //  21000 GWEI
                    value: ethers.utils.parseEther(amount)._hex,
                }]
            });
            const transactionHash = await transactionContract.addRecord(addressTo, amount, message, keyword);
            setIsTransacting(true);
            console.log(`Transacting - ${transactionHash.hash}`);
            await transactionHash.wait();   //  交易完成回调
            setIsTransacting(false);
            console.log(`Transaction Success - ${transactionHash.hash}`);
            const transactionCount = await transactionContract.getRecordCount();
            setTransactionCount(transactionCount);
        } catch (err) {
            console.error(err);
            throw new Error("No ethereum object.");
        }
    };
    const getTransactionRecords = async () => {
        try {
            if (!ethereum) {
                return alert("Please install metamask");
            }
            const {transactionContract} = getContacts();
            const transactionRecords = await transactionContract.getRecord();
            setTransactionRecords(transactionRecords.map((transaction: any) => ({
                addressTo: transaction.receiver,
                from: transaction.sender,
                amount: parseInt(transaction.amount._hex) * (10 ** 18),
                message: transaction.message,
                keyword: transaction.keyword.split(" ").join(","),
                timestamp: (new Date(transaction.timestamp.toNumber() * 1000)).toLocaleString()
            })));
        } catch (err) {
            console.error(err);
            throw new Error("No ethereum object.");
        }
    };
    useEffect(() => {
        void initWalletConnect();
    }, []);
    return (
        <TransactionContext.Provider value={{
            isTransacting,
            connectWallet,
            currentWalletAccount,
            sendTransaction,
            transactionCount,
            transactionRecords
        }}>
            {children}
        </TransactionContext.Provider>
    )
}
