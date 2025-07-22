@startuml
skinparam dpi 200
skinparam monochrome true
skinparam shadowing false

actor PaddleExecutor
participant "CommContextManager" as CCM
participant "XCCLCommContext" as XCCL
participant "CustomDevice" as Device
participant "IntelHPURuntime" as Runtime
participant "HCCL" as HCCL
participant "IntelHPU" as HPU

== Initialization ==
PaddleExecutor -> Device: CCLCommInitRank(num_ranks, unique_id, rank, &comm)
Device -> Runtime: XcclCommInitRank(...)
Runtime -> HCCL: hcclCommInitRank(...)
HCCL --> Runtime: Return hcclComm_t
Runtime --> Device: Return CCLComm
Device --> PaddleExecutor: Return CCLComm
PaddleExecutor -> CCM: GetOrCreate(ring_id)
CCM --> PaddleExecutor: Return XCCLCommContext

== Communication ==
PaddleExecutor -> CCM: ParseDeviceContext(op, place)
CCM --> PaddleExecutor: Return XCCLCommContext
PaddleExecutor -> XCCL: AllReduce(data, comm, stream)
XCCL -> Device: XcclAllReduce(data, comm, stream)
Device -> Runtime: XcclAllReduce(...)
Runtime -> HCCL: hcclAllReduce(...)
HCCL -> HPU: Execute AllReduce
HCCL --> Runtime: Return success
Runtime --> Device: Return success
Device --> XCCL: Return success
XCCL --> PaddleExecutor: Return success

@enduml