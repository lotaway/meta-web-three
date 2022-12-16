import {useState, useEffect, ChangeEvent} from "react";
import {useDebounce} from "../../utils/hooks";

type Props = {
    defaultValue?: number
};

export default function PayContract({defaultValue}: Props) {
    const [value, setValue] = useState(defaultValue ?? 0);
    const debounceValue = useDebounce(value);
    const handlerChange = (event: ChangeEvent<HTMLInputElement>) => {
        setValue(Number(event.target.value));
    };
    useEffect(() => {
        // do some api check
    }, [debounceValue]);
    return (
        <input type="number" step="0.0001" value={value} onChange={event => handlerChange(event)}/>
    )
}