The **execution substrate** packages a local, multi-repository workspace so it can be reconstructed and run on remote machines. It lets developers launch uncommitted or experimental code without pushing every change to source control or manually copying files.

`polyrepo.yaml` defines the participating repositories, their local locations, remote directory structure, and launch storage policy. During publication, the substrate creates:

* A reusable **base archive** containing a frozen workspace snapshot.
* A smaller **patch archive** containing the current tracked and visible untracked changes.
* **Bootstrap shell code** that downloads both archives, reconstructs the workspace, and prepares it for execution.

Launchers embed this bootstrap code into task scripts, then add dependency setup and workload commands. The substrate handles workspace transfer and reconstruction, while launchers and backend systems handle task planning, scheduling, resources, queues, and job lifecycle. This repository also includes generic launch helpers and a TPU dispatch CLI; project-specific bucket policy stays in files beside the workspace manifest.

The broader launch architecture has four layers:

1. **Planning:** Defines tasks, parameters, stages, and dependencies.
2. **Execution shaping:** Selects or modifies which tasks become launch requests.
3. **Launch orchestration:** Publishes the workspace, creates scripts, and stores them.
4. **Submission:** Sends those scripts to local, dispatch, or cluster backends.

The same process can support one-off diagnostic commands, keeping operator tools and larger workloads on one execution path.

The launch executor exports `POLYREPO_EXP`, `POLYREPO_MINOR`, and
`POLYREPO_PATCH` for each task, alongside
`WANDB_NAME=v{exp}.{minor}.{patch}_{cell}`. These identify the experiment
plan, its parameter branch, and the launch of that branch. It also exports
`WANDB_TAGS=v{exp},v{exp}.{minor},v{exp}.{minor}.{patch},v{exp}.X.{patch}`
so W&B can group runs by experiment, branch, branch and patch together,
or patch across all branches. Workloads can read these exports directly
to record launch metadata. After `wandb.init()`, call
`polyrepo_launch.wandb_utils.deprecate_previous_patches(run)` to tag all
existing lower patches of the same experiment, branch, and cell as
`deprecated`, preserving their other tags.

The executor also exports `POLYREPO_SHORT_CONFIG`, a JSON object containing
every knob's short name and resolved value, including defaults and knobs
inside `.named(...)` groups. After `wandb.init()`, call
`polyrepo_launch.wandb_utils.add_short_names_to_config(run)` to add these
as config keys such as `lr`, `mult`, and `model`. Numeric values stay numeric.

Only source-relevant files are be published. Ignored files—such as datasets, credentials, checkpoints, generated outputs, and machine-specific artifacts—remain outside the archives. Because patches may contain unpublished code or configuration, they require controlled storage and source-code-level security. Generated scripts can contain the W&B key, so keep the GCS artifact bucket private.

Overall, the substrate creates a shared contract between development and remote execution: declare the workspace once, publish its current state at launch time, reconstruct it remotely, and submit the resulting scripts through any supported execution backend.
