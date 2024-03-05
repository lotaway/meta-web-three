import {Link} from "react-router-dom"

export default function Index() {
    const goods = [{
        id: 462,
        name: "Test goods"
    }]
    return (
        <div className="shop-index">
            <h1 className="main-color">Shop</h1>
            {
                goods.map(item => (
                    <Link to={`/shop/goods/${item.id}`} key={item.id}>{item.name}</Link>
                ))
            }
        </div>
    )
}