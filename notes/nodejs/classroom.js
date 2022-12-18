/**
 * 导出方法
 */
var classroom = {
    students: []
};

module.exports.addTeacher = function (t) {
    classroom.teacher = t;
    console.log("the theacher is: " + classroom.teacher);
};

module.exports.addStudent = function (s) {
    s.forEach(function (index, each) {
        classroom.students[index] = each[index];
    });

    return classroom.students;
};

//内部方法
function clearAll() {
    classroom = {};
}

function add() {
    //实例方法
    this.addT = function (t) {
        classroom.teacher = t;
    };
    this.addS = function (s) {
        s.forEach(function (each) {
            classroom.students.push(each);
        });

        return classroom.students;
    };
}

//  可以通过以下语句把内部方法暴露给外部
module.exports.clearAll = clearAll;

//  可以重新声明接口为内部方法对象，以调用实例方法
module.exports = add;