class ClassRoom {
    name: string;

    constructor(private readonly _classNumber: number, className?: string) {
        this.name = className;
    }

    showClassInfo() {
        console.log(`The member value is defined by constructor private param: ${this._classNumber}`);
        // console.log(`You can't get it without private: ${this.className}`);  // NO EXIST!
        console.log(`Param without private only can receive by 'this.name': ${this.name}`);
    }
}