/**
 * Created by lotaway on 2016/6/26.
 */
//    export 导出
export const sqrt = Math.sqrt;
export class MyClass {

}
export function square(x) {
    return x * x;
}
export function diag(x, y) {
    return sqrt(square(x) + square(y));
}

//  为了方便，可以导出一个默认的内容
export default function () {
    return 'default function';
}