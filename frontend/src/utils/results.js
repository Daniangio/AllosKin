import { deleteResult } from '../api/jobs';

export async function confirmAndDeleteResult(
  jobId,
  { onSuccess, onError, confirmMessage } = {}
) {
  if (!jobId) return false;
  const message = confirmMessage || 'Delete this result?';
  if (!window.confirm(message)) return false;
  try {
    await deleteResult(jobId);
    if (onSuccess) {
      await onSuccess();
    }
    return true;
  } catch (err) {
    if (onError) {
      onError(err);
    } else {
      throw err;
    }
    return false;
  }
}
