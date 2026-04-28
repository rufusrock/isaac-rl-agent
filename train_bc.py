from __future__ import annotations

from binding_rl_agent.training import train_behavior_cloning


def main() -> None:
    result = train_behavior_cloning()
    print(f"num_samples={result.num_samples}  model={result.model_path}")
    print(f"train_loss={result.final_train_loss:.4f}  val_loss={result.final_val_loss:.4f}")
    print(f"movement_acc={result.final_val_movement_accuracy:.3f}")
    print(f"shooting_acc={result.final_val_shooting_accuracy:.3f}")
    print(f"joint_acc={result.final_val_joint_accuracy:.3f}")


if __name__ == "__main__":
    main()
