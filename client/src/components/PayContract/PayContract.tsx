import {useState, useEffect, ChangeEvent, useTransition} from "react";
// import {useDebounce} from "../../utils/hooks";

type Props = {
    defaultValue?: number
};

export default function PayContract({defaultValue}: Props) {
    const [value, setValue] = useState(defaultValue ?? 0);
    const [isLoading, startTransition] = useTransition();
    const handlerChange = (event: ChangeEvent<HTMLInputElement>) => {
        startTransition(() => {
            // do some api check maybe ?
            setValue(Number(event.target.value));
        });
    };
    useEffect(() => {
        value && handlerChange({
            target: {
                value: value.toString()
            }
        } as ChangeEvent<HTMLInputElement>);
    }, [value]);
    return (
        <>
            {
                isLoading ? <p>Loading...</p> : null
            }
            <input type="number" step="0.0001" value={value} onChange={event => handlerChange(event)}/>
        </>
    );
}