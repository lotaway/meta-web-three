import {useCallback, useContext, useEffect, useMemo, useState} from "react";
import {TransactionContext} from "../../context/TransactionContext";
import PayContract from "../../components/PayContract/PayContract";
import {useDebounce} from "../../utils/hooks";
// 来源：https://github.com/adrianhajdin/project_web3.0
export default function TransactionRecord() {
    const {isTransacting, transactionRecords, transactionCount} = useContext(TransactionContext);
    const [addressTo, setAddressTo] = useState("");
    const [_accountName, setAccountName] = useState(addressTo);
    const accountName = useDebounce(_accountName);
    return (
        <div className="w-full">
            <label className="">Please input a address：</label>
            <input type="text" value={_accountName} onChange={event => setAccountName(event.target.value)}/>
            <button type="button" onClick={() => setAddressTo(_accountName)}>Confirm</button>
            {addressTo ? <PayContract addressTo={addressTo}/> : null}
            {isTransacting ? "transacting..." : null}
            <div className="bg-black transaction-records">
                {
                    transactionRecords.map((item: {
                        addressTo: string
                        from: string
                        amount: number
                        message: string
                        keyword: string
                        timestamp: string
                    }) => (
                        <div className="transaction-record-info">
                            <h3>{item.addressTo}</h3>
                            <p>{item.from}</p>
                            <p>{item.amount}</p>
                            <p>{item.message}</p>
                            <p>{item.keyword}</p>
                            <p>{item.timestamp}</p>
                        </div>
                    ))
                }
            </div>
        </div>
    )
}