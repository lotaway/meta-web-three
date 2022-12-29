import React, {useState, useEffect, ChangeEvent, useTransition, useContext} from "react";
import {TransactionContext} from "../../context/TransactionContext";
// import {useDebounce} from "../../utils/hooks";

type Props = {
    amount?: string
};

export default function PayContract({amount}: Props) {
    const {currentWalletAccount, connectWallet, sendTransaction} = useContext(TransactionContext);
    const [formData, setFormData] = useState({
        addressTo: "",
        amount: amount ?? "0",
        keyword: "",
        message: ""
    });
    const [isLoading, startTransition] = useTransition();
    const handleChange = (event: ChangeEvent<HTMLInputElement>, name: string) => {
        startTransition(() => {
            // do some api check maybe ?
            // setValue(Number(event.target.value));
            setFormData(prevState => ({...prevState, [name]: event.target?.value || ""}));
        });
    };
    const handleSubmit = (event: any) => {
        event.preventDefault();
        const {addressTo, amount, keyword, message} = formData;
        if (!addressTo || !amount || !keyword || !message) {
            return;
        }
        sendTransaction({
            addressTo,
            amount,
            keyword,
            message
        });
    };
    return (
        <form>
            {
                isLoading ? <p>Loading...</p> : null
            }
            {!currentWalletAccount ? <button onClick={connectWallet}>Connect Wallet</button> : null}
            <input type="number" step="0.0001" value={formData.amount}
                   onChange={event => handleChange(event, "amount")}/>
            <button type="submit" onClick={event => handleSubmit(event)}></button>
        </form>
    );
}