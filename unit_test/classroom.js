/**
 * Created by lotaway on 2016/4/1.
 */
var classroom = {};

exports.addTeacher = function (t) {
    classroom.teacher = t;
    console.log("the theacher is: " + classroom.teacher);
};

exports.addStudent = function (s) {
    s.forEach(function (index, each) {
        classroom.students[index] = each[index];
    });
    console.log(classroom.students);
};
//内部方法
function clearAll() {
    classroom = {};
}
function add() {
    //实例方法
    this.addT = function (t) {
        classroom.teacher = t;
        console.log(classroom.teacher);
    };
    this.addS = function (s) {
        s.forEach(function (index, each) {
            classroom.students[index] = each[index];
        });
        console.log(classroom.students);
    };
}

//  可以通过以下语句把内部方法暴露给外部
exports.clearAll = clearAll;

//  可以重新声明接口为内部方法对象，以调用实例方法
module.exports = add;