const readline = require("readline"), insReadline = readline.createInterface({
    input: process.stdin, output: process.stdout
});

/**
 * 判圈算法找到数组里的重复项（判圈算法又称龟兔赛跑算法）
 * @param {Array<number>} nums 需要找出重复项的数组
 */

function findDuplicate(nums) {
    let tortoise = 0, hare = 0;
    while (tortoise !== hare && hare !== undefined) {
        tortoise = nums[tortoise];
        hare = nums[hare];
        if (hare !== undefined) {
            hare = nums[hare];
        }
    }
    if (hare === undefined) return hare;
    tortoise = 0;
    while (tortoise !== hare) {
        tortoise = nums[tortoise];
        hare = nums[hare];
    }
    return hare;
}

function hashMap(nums) {
    let map = new Map(), startIndex = 0, value = null;
    while (true) {
        value = nums[startIndex];
        map.set(value, true);
        if (map.has(value)) break;
    }
    return value;
}

insReadline.question("If input a array such as `1,4,5,2,2`, will output `2`：", input => {
    console.log(input);
    try {
        console.log(findDuplicate(input.split(",")));
    } catch (err) {
        console.log("Error: " + JSON.stringify(err));
    }
});