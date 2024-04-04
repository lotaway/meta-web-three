export namespace ClientStore {

    function test() {
        const upgradeDBScript = new UpgradeDBScript(1)
        upgradeDBScript.addCommand(new ClientDBTable("nfts"))
    }

    export class ClientDB {

        protected upgradeScripts: UpgradeDBScript[] = []

        constructor(readonly dbFactory: IDBFactory) {

        }

        addUprade(script: UpgradeDBScript) {
            this.upgradeScripts.push(script)
        }

        connect(dbName: string, version: number = 1) {
            const self = this
            const request = this.dbFactory.open(dbName, version)
            return new Promise(function (resolve, reject) {
                request.onupgradeneeded = function (event) {
                    const db = request.result
                    self.upgradeScripts.sort((first, second) => first.version - second.version).map(script => {
                        if (event.oldVersion < script.version) {
                        }
                    })
                }
                request.onsuccess = function (event) {
                    resolve(request.result)
                }
                request.onerror = reject
            })
        }

    }

    export class UpgradeDBScript {

        protected commands: DBCommand[] = []

        constructor(readonly version: number) {

        }

        addCommand(command: DBCommand) {
            this.commands.push(command)
        }

        use(db: IDBDatabase) {
            for (const command of this.commands) {
                command.use(db)
            }
        }

    }

    export interface DBCommand {
        use(db: IDBDatabase): void
    }

    export interface IPageWrapper<Params extends object> {
        getParams(): Params
        getPage(): number
        getPageSize(): number
        getRange(): IDBKeyRange
    }

    export abstract class BasePageWrapper<VO extends object> implements IPageWrapper<VO> {
        getPage(): number {
            return 1
        }
        getPageSize(): number {
            return 20
        }
        getRange(): IDBKeyRange {
            return IDBKeyRange.only('id')
        }
        abstract getParams(): VO
    }

    export class ClientDBTable<DO extends object> implements DBCommand {

        protected fields: ClientDBField[] = []

        constructor(readonly name: string, readonly key: string = "id") {

        }

        addField(field: ClientDBField) {
            if (this.key === field.name) {
                throw new Error(`New field can't have the same as key name: ${this.key}`)
            }
            this.fields.push(field)
        }

        use(db: IDBDatabase): IDBObjectStore {
            const objectStore = db.createObjectStore(this.name, {
                keyPath: this.key,
            })
            for (const field of this.fields) {
                field.use(objectStore)
            }
            return objectStore
        }

        getData(db: IDBDatabase) {
            const transaction = db.transaction(this.name, "readonly")
            const store = transaction.objectStore(this.name)
            const request = store.getAll()
            return new Promise((resolve, reject) => {
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        getDataByPage(store: IDBObjectStore, wrapper: IPageWrapper<Partial<DO>>) {
            const page = wrapper.getPage()
            const pageSize = wrapper.getPageSize()
            const request = store.openCursor(wrapper.getRange())
            return new Promise((resolve, reject) => {
                const tableRows: DO[] = []
                let needSkip = true
                request.onsuccess = function (event) {
                    const cursor = request.result
                    if (!cursor || !cursor.value) {
                        return resolve(tableRows)
                    }
                    if (needSkip) {
                        cursor.advance((page - 1) * pageSize)
                        needSkip = false
                    }
                    if (tableRows.length < pageSize) {
                        return cursor.continue()
                    }
                    return resolve(tableRows)
                }
                request.onerror = reject
            })
        }

        addData(store: IDBObjectStore, data: DO) {
            return new Promise((resolve, reject) => {
                const request = store.add(data)
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        updateData(store: IDBObjectStore, data: DO) {
            return new Promise((resolve, reject) => {
                const request = store.put(data)
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        delData(store: IDBObjectStore, key: string) {
            return new Promise((resolve, reject) => {
                const request = store.delete(IDBKeyRange.only(key))
                request.onsuccess = resolve
                request.onerror = reject
            })
        }
    }

    export class ClientDBField<Type = string> {

        constructor(readonly data: {
            name: string
            key?: string
            unique?: boolean
        }) {
            this.data.unique = this.data.unique ?? false
        }

        get name() {
            return this.data.name
        }

        use(objectStore: IDBObjectStore) {
            objectStore.createIndex(this.data.name, this.data.key ?? this.data.name, {
                unique: this.data.unique,
            })
        }

        updateData(data: Type) {
            // @todo
        }
    }
}