import {
    forwardRef,
    ButtonHTMLAttributes,
    FunctionComponent,
    PropsWithChildren,
    PropsWithRef,
    LegacyRef
} from "react"
import style from "./AntButton.module.less"

type IAntButtonType = "default" | "primary" | "dashed" | "text" | "link"

// ButtonHTMLAttributes<HTMLButtonElement>
type IProps = PropsWithRef<PropsWithChildren<{
    ref?: LegacyRef<HTMLButtonElement>
    type?: IAntButtonType
    icon?: string
    shape?: string
    className: ButtonHTMLAttributes<any>["className"]
}>>

type InferFirst<T> = T extends [infer First, ...any[]] ? First : T

function getItem<ItemType>(args: [ItemType, ...unknown[]] | ItemType): ItemType {
    return Array.isArray(args) ? args[0] : args
}

function handler<ValueType>(value: ValueType) {

    const result = getItem([3, "2", false])
    const result2 = getItem(value)
    //  TODO 希望拿到的result类型是3的类型，即number
    type Result = InferFirst<[number, string, boolean]>
    // console.log(result, result2)
}


const AntButton: FunctionComponent<IProps> = ({ref, type, shape, className, children, ...rProps}) => {
    switch (type) {
        case style.primary:
        case style.dashed:
        case style.text:
        case style.link:
            // console.log("match")
            break
        default:
            break
    }
    const styles: string = (className ?? "") + (type ?? style.primary) + (shape ?? "")
    return (
        <button ref={ref} className={styles} {...rProps}>{children}</button>
    )
}

// export default forwardRef<IProps>(AntButton)