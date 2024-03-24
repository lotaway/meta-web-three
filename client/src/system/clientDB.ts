export namespace ClientStore {
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
                request.onsuccess = function(event) {
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

        static test() {
            const upgradeDBScript = new UpgradeDBScript(1)
            new ClientDBTable("nfts")
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

    export class ClientDBTable implements DBCommand {

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
            return db.createObjectStore(this.name, {
                keyPath: this.key,
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