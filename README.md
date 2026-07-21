The **execution substrate** packages a local, multi-repository workspace so it can be reconstructed and run on remote machines. It lets developers launch uncommitted or experimental code without pushing every change to source control or manually copying files.

A workspace manifest defines the participating repositories, their local locations, and their remote directory structure. During publication, the substrate creates:

* A reusable **base archive** containing a frozen workspace snapshot.
* A smaller **patch archive** containing the current tracked and visible untracked changes.
* **Bootstrap shell code** that downloads both archives, reconstructs the workspace, and prepares it for execution.

Launchers embed this bootstrap code into task scripts, then add dependency setup and workload commands. The substrate handles workspace transfer and reconstruction, while launchers and backend systems handle task planning, scheduling, resources, queues, and job lifecycle.

The broader launch architecture has four layers:

1. **Planning:** Defines tasks, parameters, stages, and dependencies.
2. **Execution shaping:** Selects or modifies which tasks become launch requests.
3. **Launch orchestration:** Publishes the workspace, creates scripts, and stores them.
4. **Submission:** Sends those scripts to local, dispatch, or cluster backends.

The same process can support one-off diagnostic commands, keeping operator tools and larger workloads on one execution path.

Only source-relevant files are be published. Ignored files—such as datasets, credentials, checkpoints, generated outputs, and machine-specific artifacts—remain outside the archives. Because patches may contain unpublished code or configuration, they require controlled storage and source-code-level security.

Overall, the substrate creates a shared contract between development and remote execution: declare the workspace once, publish its current state at launch time, reconstruct it remotely, and submit the resulting scripts through any supported execution backend.
