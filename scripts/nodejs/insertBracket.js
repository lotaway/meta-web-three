/*
# 根据输入的数字n(1<n<8)作为括号总数量，并输出所有括号可能性组合的数组string[]。
## 例子1
输入：n=1
输出：["()"]
## 例子2
输入：n=3
输出：["((()))", "(()())", "(())()", "()(())", "()()()"]
*/
const readline = require("readline");
const insReadline = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function showQuestion(question = "Please input the number of brackets you want, 0<number<8 \n", _handler = handler) {
    insReadline.question(question, input => {
        try {
            if (input >= 8 || input < 1) {
                throw new Error("input should be in [1,8].");
            }
            console.log("result:" + JSON.stringify(_handler(input)));
        } catch (err) {
            console.log(err);
        }
        console.log("End...");
        showQuestion();
    });
}

/*
第一种思路是打散字符串插值
将字符串A插入到长度为L的字符串B中所有字符的前后间隙位置中得到Array(L+1)个可能性结果，之后再将结果去重得到本次真正的结果。
之后根据输入值循环n次，得到最终结果。
其中字符串A="()"，字符串B就是上一次得到的n-1结果数组[字符串B1,字符串B2]，进行迭代得到的。
 */
function handler(n) {
    //  code here...
    let res = [""];
    //  获取所有插值可能性
    const single = (str = "", ins = "()") => {
        const strArr = [...str.split(""), ""];  //  数组最后一位添加任意项，确保前后空隙都会插入值
        return Array.from(new Set(strArr.map((noMatter, index) => {
            let _arr = Array.from(strArr);
            _arr.splice(index, 0, ins);
            return _arr.join("");
        })));
    };
    //  根据输入值执行n次插入
    while (n > 0) {
        res = res.reduce((prev, item) => Array.from(new Set([...prev, ...single(item)])), []);
        n--;
    }
    return res;
}
/*
第二种思路是先转换为数值单位进行叠加
是先将字符值"()"视为一个数值单位1，之后每次进行是间隙和自身都叠加值，第一次是[1]，第二次是[[1,1],[2]]，第三次是[[1,1,1],[2,1],[1,2],[3]]，最后才根据单位数量叠加数量重新替换为字符值，例如2="(())"，3="((()))"。
本质上这种方法没有脱离第一种思路，只是少了迭代过程中数组和字符串之间频繁来回转换的行为，只需要在首尾各进行一次转换即可，所以不再放具体代码，感兴趣的可以自行实现。
 */

//  don't touch
showQuestion();