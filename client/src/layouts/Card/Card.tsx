import {PropsWithChildren} from "react"

interface CardProps extends PropsWithChildren {
}

interface CardHeaderProps extends PropsWithChildren {
}

interface CardBodyProps extends PropsWithChildren {
}

interface CardBelowProps extends PropsWithChildren {
    secTitle?: string
}

export default function Card({children}: CardProps) {
    return (
        <section className="layout-card">
            {children}
        </section>
    )
}
Card.Header = function ({children}: CardHeaderProps) {
    return (
        <h1 className="layout-card-title text-3xl font-bold underline">{children}</h1>)
}
Card.Body = function (props: CardBodyProps) {
    return (
        <div className="layout-card-body">
            {props.children}
        </div>
    )
}
Card.Below = function ({children}: CardBelowProps) {
    return (
        <div className="layout-card-below">
            {children}
        </div>
    )
}
