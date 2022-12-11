// https://jestjs.io/docs/en/getting-started
const Add = require("../nodejs/classroom");

test("判断是否成功添加学生", () => {
    const arr = (new Add).addS(["Tom", "Jelly", "Jam"]);
    let n;

    expect(arr[arr.length - 1]).toBe("Jam");
    expect(arr[arr.length - 2]).not.toBe("Jam");  //  判断不等
    expect(null).toBeNull();   //    判断是否为null
    expect(n).toBeUndefined();  //   判断是否为undefined
    expect(arr).toBeDefined();    //  判断结果与toBeUndefined相反
    expect(arr.length).toBeTruthy(); //  判断结果为true
    expect(typeof arr === "string").toBeFalsy();  //  判断结果为false
    expect(arr.length).toBeGreaterThan(2);   //    大于2
    expect(arr.length).toBeGreaterThanOrEqual(3);  //  大于等于3.5
    expect(arr.length).toBeLessThan(5);  //   小于5
    expect(arr.length).toBeLessThanOrEqual(3); //  小于等于4.5
    expect(arr.length + 0.5).toBeCloseTo(3.5); //  浮点数判断相等
    expect(arr[arr.length - 1]).toMatch(/Ja/);    // 正则判断
    expect(arr).toContain('Tom');    //  是否包含
});